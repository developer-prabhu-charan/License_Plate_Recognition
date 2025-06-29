# 3_recognize_characters.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
from ultralytics import YOLO # Assuming you trained with Ultralytics YOLO for detection
import pandas as pd # Import pandas for CSV output

# --- Paths ---
YOLO_MODEL_PATH = "/content/drive/MyDrive/LPR_Project/3_models/yolo_plate_detector/weights/best.pt"
OCR_MODEL_PATH = "/content/drive/MyDrive/LPR_Project/3_models/ocr_recognizer/best_ocr_model.pth"
CHAR_LIST_PATH = "/content/drive/MyDrive/LPR_Project/1_data/recognition_ocr_dataset/char_list.txt"
TEST_IMAGES_DIR = "/content/drive/MyDrive/LPR_Project/1_data/test/images"
OUTPUT_DIR = "/content/drive/MyDrive/LPR_Project/5_output/recognized_plates"
SUBMISSION_FILE_PATH = os.path.join(OUTPUT_DIR, "SampleSubmission.csv") # Output to CSV

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- CRNN Model Definition (Must be identical to 2_train_ocr.py) ---
class CRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=256, rnn_num_layers=2):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64x32x128
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128x16x64
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0,1)), # Output: 256x8x64
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0,1)), # Output: 512x4x64
            
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1, padding=0),
            nn.BatchNorm2d(512), nn.ReLU(True) # Output: 512x1x64
        )
        
        self.map_to_sequence = nn.Linear(512, rnn_hidden_size)

        self.rnn = nn.LSTM(input_size=rnn_hidden_size, 
                           hidden_size=rnn_hidden_size, 
                           num_layers=rnn_num_layers,
                           bidirectional=True, 
                           batch_first=False)

        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, input):
        conv_output = self.cnn(input) # (batch, features, 1, W_prime)
        
        conv_output = conv_output.squeeze(2) # (batch, features, W_prime)
        conv_output = conv_output.permute(2, 0, 1) # (W_prime, batch, features) -> (seq_len, batch, input_size)
        
        rnn_input = self.map_to_sequence(conv_output)
        recurrent_features, _ = self.rnn(rnn_input)
        output = self.fc(recurrent_features)
        return output

# --- CTC Decoder (Greedy) ---
def ctc_decoder(log_probs, idx_to_char):
    max_probs = log_probs.argmax(dim=-1)
    
    decoded_chars = []
    prev_char_idx = -1
    for char_idx in max_probs:
        if char_idx.item() == len(idx_to_char) - 1: # Check if it's the CTC_BLANK token (last index)
            prev_char_idx = char_idx.item()
            continue
        if char_idx.item() != prev_char_idx:
            decoded_chars.append(idx_to_char[char_idx.item()])
        prev_char_idx = char_idx.item()
            
    return "".join(decoded_chars)

# --- Main Recognition Pipeline ---
def recognize_license_plates(yolo_model, ocr_model, char_to_idx, idx_to_char, test_images_dir, output_dir, submission_file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ocr_model.to(device)
    ocr_model.eval()

    ocr_transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    results_data = [] # To store data for DataFrame

    for img_name in tqdm(os.listdir(test_images_dir), desc="Processing Test Images"):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            continue

        img_path = os.path.join(test_images_dir, img_name)
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(original_img_rgb)

        detection_results = yolo_model(pil_img, verbose=False)

        recognized_plate_text = "No_Plate_Detected" # Default if no plate is found

        # Process detection results - take the first detected plate if multiple
        if detection_results and detection_results[0].boxes:
            box = detection_results[0].boxes[0] # Take the first detected box (highest confidence)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]

            if conf > 0.5:
                y1, x1, y2, x2 = max(0, y1), max(0, x1), min(original_img.shape[0], y2), min(original_img.shape[1], x2)
                
                cropped_plate_pil = pil_img.crop((x1, y1, x2, y2))

                if cropped_plate_pil.width > 0 and cropped_plate_pil.height > 0:
                    ocr_input = ocr_transform(cropped_plate_pil).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        ocr_output = ocr_model(ocr_input)
                    
                    ocr_output = ocr_output.squeeze(1)
                    
                    recognized_plate_text = ctc_decoder(ocr_output, idx_to_char)
                else:
                    recognized_plate_text = "[Invalid_Crop]"

                # Draw bounding box and text on the original image for visualization
                cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(original_img, recognized_plate_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save annotated image
        output_img_path = os.path.join(output_dir, f"recognized_{img_name}")
        cv2.imwrite(output_img_path, original_img)

        # Append result for CSV
        results_data.append({'id': img_name, 'text': recognized_plate_text})

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(submission_file_path, index=False)

    print(f"\nProcessing complete! Annotated images saved to: {output_dir}")
    print(f"Recognized plate numbers (id,text) saved to: {submission_file_path}")


if __name__ == "__main__":
    # --- Load Character List (Vocabulary) ---
    if not os.path.exists(CHAR_LIST_PATH):
        print(f"Error: char_list.txt not found at {CHAR_LIST_PATH}. Please run 1_prepare_ocr_data.py first.")
        exit()

    with open(CHAR_LIST_PATH, 'r') as f:
        char_list = [char.strip() for char in f.readlines()]
    char_list.append('[CTC_BLANK]') # Must match how it was added during training
    char_to_idx = {char: i for i, char in enumerate(char_list)}
    idx_to_char = {i: char for i, char in enumerate(char_list)}
    num_classes = len(char_list)

    print(f"OCR Vocabulary loaded with {num_classes} characters.")

    # --- Load YOLO Detection Model ---
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}. Please ensure your detection model is trained and saved.")
        exit()
    print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("YOLO model loaded.")

    # --- Load OCR Recognition Model ---
    ocr_model = CRNN(num_classes=num_classes)
    if not os.path.exists(OCR_MODEL_PATH):
        print(f"Error: OCR model not found at {OCR_MODEL_PATH}. Please ensure your OCR model is trained and saved.")
        exit()
    print(f"Loading OCR model from {OCR_MODEL_PATH}...")
    try:
        ocr_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading OCR model state_dict: {e}")
        print("Please ensure the CRNN class definition in this script is IDENTICAL to the one used for training.")
        exit()
    print("OCR model loaded.")
    
    # --- Run the full pipeline ---
    recognize_license_plates(yolo_model, ocr_model, char_to_idx, idx_to_char, TEST_IMAGES_DIR, OUTPUT_DIR, SUBMISSION_FILE_PATH)