# 1_prepare_ocr_data.py
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import string
import cv2 # For dummy image creation
import numpy as np # For dummy image creation

def prepare_ocr_dataset(
    csv_path,
    img_dir,
    output_base_dir,
    test_size=0.2,
    random_state=42,
    img_id_col='img_id', # Adjust this if your CSV column name is 'image_id'
    text_col='text'
):
    """
    Prepares data for OCR training by creating train/val manifest files
    and a character list.

    Args:
        csv_path (str): Path to the input CSV with img_id and text annotations.
        img_dir (str): Directory containing the license plate images.
        output_base_dir (str): Base directory where the OCR dataset structure
                                (train.txt, val.txt, char_list.txt) will be created.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
        img_id_col (str): The column name in the CSV for image IDs.
        text_col (str): The column name in the CSV for license plate text.
    """
    df = pd.read_csv(csv_path)

    # Create output directories
    os.makedirs(output_base_dir, exist_ok=True)
    output_images_dir = os.path.join(output_base_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)

    train_output_path = os.path.join(output_base_dir, 'train.txt')
    val_output_path = os.path.join(output_base_dir, 'val.txt')
    char_list_path = os.path.join(output_base_dir, 'char_list.txt')

    # Split dataframe into train and validation sets based on image IDs
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    print(f"Total annotations: {len(df)}")
    print(f"Train annotations: {len(train_df)}")
    print(f"Validation annotations: {len(val_df)}")

    all_characters = set()
    
    # Process training data
    print("Processing training data...")
    with open(train_output_path, 'w') as f_train:
        for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train OCR"):
            img_id = row[img_id_col]
            text_label = str(row[text_col]).strip()
            
            img_full_path = os.path.join(img_dir, img_id)
            
            if not os.path.exists(img_full_path):
                print(f"Warning: Image file not found: {img_full_path}. Skipping.")
                continue

            f_train.write(f"{img_id} {text_label}\n")
            
            shutil.copy(img_full_path, os.path.join(output_images_dir, img_id))

            for char in text_label:
                all_characters.add(char)

    # Process validation data
    print("Processing validation data...")
    with open(val_output_path, 'w') as f_val:
        for index, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val OCR"):
            img_id = row[img_id_col]
            text_label = str(row[text_col]).strip()
            
            img_full_path = os.path.join(img_dir, img_id)
            
            if not os.path.exists(img_full_path):
                print(f"Warning: Image file not found: {img_full_path}. Skipping.")
                continue
            
            f_val.write(f"{img_id} {text_label}\n")
            
            shutil.copy(img_full_path, os.path.join(output_images_dir, img_id))

            for char in text_label:
                all_characters.add(char)

    # Create character list (vocabulary)
    sorted_characters = sorted(list(all_characters))
    with open(char_list_path, 'w') as f_chars:
        for char in sorted_characters:
            f_chars.write(char + '\n')

    print(f"OCR dataset prepared at: {output_base_dir}")
    print(f"Train manifest: {train_output_path}")
    print(f"Validation manifest: {val_output_path}")
    print(f"Character list (vocabulary): {char_list_path} (Contains {len(sorted_characters)} unique characters)")

if __name__ == "__main__":
    # --- IMPORTANT: Paths updated as per your specification ---
    csv_input_path = "/content/drive/MyDrive/LPR_Project/1_data/train_ocr/labels.csv"
    images_input_dir = "/content/drive/MyDrive/LPR_Project/1_data/train_ocr/images"
    # Output directory for the prepared OCR dataset
    ocr_output_base_dir = "/content/drive/MyDrive/LPR_Project/1_data/recognition_ocr_dataset" 

    # --- Create dummy data for demonstration if not already present ---
    # REMOVE OR COMMENT OUT THIS BLOCK IF YOU HAVE YOUR ACTUAL DATA READY
    # AND DON'T WANT DUMMY DATA TO BE CREATED.
    if not os.path.exists(csv_input_path):
        print("Creating dummy OCR data for preparation demonstration...")
        os.makedirs(images_input_dir, exist_ok=True)
        dummy_data = {
            'img_id': [f'lp_{i:04d}.jpg' for i in range(1, 901)],
            'text': [f'AB12CD{i:04d}' for i in range(1, 901)] # Example plate numbers
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(csv_input_path, index=False)

        for i in range(1, 901):
            img_path = os.path.join(images_input_dir, f'lp_{i:04d}.jpg')
            dummy_img = np.zeros((60, 120, 3), dtype=np.uint8) + 200 # Light gray background
            text = f'AB12CD{i:04d}'
            cv2.putText(dummy_img, text, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imwrite(img_path, dummy_img)
        print("Dummy OCR data created.")
    # ------------------------------------------------------------------

    # Call the data preparation function
    prepare_ocr_dataset(
        csv_path=csv_input_path,
        img_dir=images_input_dir,
        output_base_dir=ocr_output_base_dir,
        img_id_col='img_id', # IMPORTANT: Ensure this matches your labels.csv header
        text_col='text'      # IMPORTANT: Ensure this matches your labels.csv header
    )
    print("OCR data preparation complete.")