# 2_train_ocr.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import random
import string

# Define paths
BASE_DATA_DIR = "/content/drive/MyDrive/LPR_Project/1_data/recognition_ocr_dataset"
TRAIN_MANIFEST = os.path.join(BASE_DATA_DIR, 'train.txt')
VAL_MANIFEST = os.path.join(BASE_DATA_DIR, 'val.txt')
CHAR_LIST_PATH = os.path.join(BASE_DATA_DIR, 'char_list.txt')
IMAGES_DIR = os.path.join(BASE_DATA_DIR, 'images')

MODEL_SAVE_PATH = "/content/drive/MyDrive/LPR_Project/3_models/ocr_recognizer/best_ocr_model.pth"
LOG_DIR = "/content/drive/MyDrive/LPR_Project/4_runs/ocr_training_logs"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# --- 1. Dataset Class ---
class OCRDataset(Dataset):
    def __init__(self, manifest_path, images_dir, char_to_idx, transform=None):
        self.images_dir = images_dir
        self.char_to_idx = char_to_idx
        self.transform = transform
        self.data = [] # List of (image_filename, text_label) tuples

        with open(manifest_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1) # Split only on the first space
                if len(parts) == 2:
                    img_filename, text_label = parts
                    self.data.append((img_filename, text_label))
                else:
                    print(f"Skipping malformed line in manifest: {line.strip()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename, text_label = self.data[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning dummy image.")
            # Return a dummy black image if loading fails or for debugging
            img = Image.fromarray(np.zeros((64, 256, 3), dtype=np.uint8))

        if self.transform:
            img = self.transform(img)

        text_seq = [self.char_to_idx[char] for char in text_label if char in self.char_to_idx]
        text_seq = torch.LongTensor(text_seq)

        return img, text_seq, len(text_label)

# Custom Collate Function for DataLoader
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    
    images = torch.stack(images, 0)
    
    max_label_len = max(label_lengths)
    padded_labels = torch.full((len(labels), max_label_len), 0, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    
    label_lengths = torch.LongTensor(label_lengths)
    
    return images, padded_labels, label_lengths


# --- 2. Model Architecture (CRNN) - IMPROVED CNN BACKBONE ---
class CRNN(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=256, rnn_num_layers=2):
        super(CRNN, self).__init__()

        # Improved CNN Backbone
        # Designed to reduce height significantly (to 1 or small value) while preserving width
        self.cnn = nn.Sequential(
            # Block 1: Input 3x64x256
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 64x32x128
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 128x16x64
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0,1)), # Output: 256x8x64 (H reduced, W preserved)
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0,1)), # Output: 512x4x64 (H reduced, W preserved)
            
            # Block 5: Final reduction to height 1
            nn.Conv2d(512, 512, kernel_size=(4, 1), stride=1, padding=0), # Conv to reduce H from 4 to 1
            nn.BatchNorm2d(512), nn.ReLU(True) # Output: 512x1x64
        )
        
        # This maps the 512 features from CNN to RNN's hidden size
        # Assuming final CNN output is (batch_size, 512, 1, ~W_prime)
        self.map_to_sequence = nn.Linear(512, rnn_hidden_size)

        # Bidirectional LSTM
        self.rnn = nn.LSTM(input_size=rnn_hidden_size, 
                           hidden_size=rnn_hidden_size, 
                           num_layers=rnn_num_layers,
                           bidirectional=True, 
                           batch_first=False) # batch_first=False for (seq_len, batch, input_size)

        # Output layer for CTC
        # Multiplied by 2 because of bidirectional RNN
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, input):
        # CNN
        conv_output = self.cnn(input) # (batch, features, 1, W_prime)
        
        # Squeeze height dimension and permute for RNN (W_prime, B, Features)
        conv_output = conv_output.squeeze(2) # (batch, features, W_prime)
        conv_output = conv_output.permute(2, 0, 1) # (W_prime, batch, features) -> (seq_len, batch, input_size)
        
        # Map CNN features to RNN input size
        rnn_input = self.map_to_sequence(conv_output)

        # RNN
        recurrent_features, _ = self.rnn(rnn_input) # (seq_len, batch, hidden_size * 2)

        # Output
        output = self.fc(recurrent_features) # (seq_len, batch, num_classes)
        return output

# --- 3. Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, char_to_idx, idx_to_char, model_save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for images, labels, label_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train"):
            images = images.to(device)

            optimizer.zero_grad()
            
            log_probs = model(images) 
            
            # Input lengths for CTC loss: sequence length of CNN output (width)
            # This is log_probs.size(0) because log_probs is (seq_len, batch_size, num_classes)
            input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long)
            
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Training Loss: {epoch_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, label_lengths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val"):
                images = images.to(device)
                
                log_probs = model(images)
                input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long)
                
                loss = criterion(log_probs, labels, input_lengths, label_lengths)
                val_loss += loss.item() * images.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch+1} Validation Loss: {epoch_val_loss:.4f}")

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} (Validation Loss: {best_val_loss:.4f})")

    print("Training finished!")


# --- 4. Main execution block ---
if __name__ == "__main__":
    # --- Dummy Data Creation for Testing (if manifests/char_list not present) ---
    # This block ensures the script can run even if 1_prepare_ocr_data.py hasn't run yet.
    # For actual training, ensure 1_prepare_ocr_data.py has been run and these files exist.
    if not os.path.exists(TRAIN_MANIFEST) or not os.path.exists(CHAR_LIST_PATH):
        print("--- Creating dummy manifest and char_list for demonstration ---")
        os.makedirs(IMAGES_DIR, exist_ok=True)
        dummy_chars = string.ascii_uppercase + string.digits
        
        with open(CHAR_LIST_PATH, 'w') as f:
            for char in sorted(list(dummy_chars)):
                f.write(char + '\n')
        
        dummy_manifest_data = []
        for i in range(1, 101):
            img_filename = f"lp_dummy_{i:03d}.jpg"
            text_label = ''.join(random.choices(dummy_chars, k=random.randint(4, 8)))
            dummy_manifest_data.append(f"images/{img_filename} {text_label}")
            
            dummy_img_path = os.path.join(IMAGES_DIR, img_filename)
            dummy_img = np.zeros((64, 256, 3), dtype=np.uint8) + 200
            cv2.putText(dummy_img, text_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imwrite(dummy_img_path, dummy_img)

        random.shuffle(dummy_manifest_data)
        train_len = int(len(dummy_manifest_data) * 0.8)
        
        with open(TRAIN_MANIFEST, 'w') as f:
            for line in dummy_manifest_data[:train_len]:
                f.write(line + '\n')
        with open(VAL_MANIFEST, 'w') as f:
            for line in dummy_manifest_data[train_len:]:
                f.write(line + '\n')
        print("--- Dummy manifest and char_list created. ---")
    # --- End Dummy Data Creation ---

    # Load character list (vocabulary)
    with open(CHAR_LIST_PATH, 'r') as f:
        char_list = [char.strip() for char in f.readlines()]
    
    # Add a blank token for CTC loss. This is typically the last index.
    char_list.append('[CTC_BLANK]') 
    char_to_idx = {char: i for i, char in enumerate(char_list)}
    idx_to_char = {i: char for i, char in enumerate(char_list)}
    num_classes = len(char_list)

    print(f"OCR Vocabulary loaded: {len(char_list)} characters.")

    # Image transformations: Resize, ToTensor, Normalize
    transform = transforms.Compose([
        transforms.Resize((64, 256)), # Fixed height and width for consistency
        transforms.ToTensor(),       # Converts to [0, 1] range, and C x H x W
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1]
    ])

    # Instantiate Datasets and DataLoaders
    train_dataset = OCRDataset(TRAIN_MANIFEST, IMAGES_DIR, char_to_idx, transform)
    val_dataset = OCRDataset(VAL_MANIFEST, IMAGES_DIR, char_to_idx, transform)

    # num_workers > 0 for faster data loading, but might need adjustment for your environment
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Model, Loss, Optimizer
    model = CRNN(num_classes=num_classes)
    criterion = nn.CTCLoss(blank=char_to_idx['[CTC_BLANK]'], zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 100 # Increased epochs for potentially better training
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, char_to_idx, idx_to_char, MODEL_SAVE_PATH)