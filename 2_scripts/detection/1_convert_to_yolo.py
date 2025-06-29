# 1_convert_to_yolo.py
import pandas as pd
import cv2
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np # Ensure numpy is imported for dummy data creation

def create_yolo_dataset(
    csv_path,
    img_dir,
    output_base_dir,
    test_size=0.2,
    random_state=42
):
    """
    Converts CSV annotations to YOLO format and structures the dataset
    into train/val splits for YOLO training.

    Args:
        csv_path (str): Path to the input CSV with img_id,xmin,ymin,xmax,ymax.
        img_dir (str): Directory containing the images.
        output_base_dir (str): Base directory where the YOLO dataset structure
                                (train/images, train/labels, val/images, val/labels)
                                will be created.
        test_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.
    """
    df = pd.read_csv(csv_path)

    # Ensure output directories exist
    train_img_dir = os.path.join(output_base_dir, 'train', 'images')
    train_label_dir = os.path.join(output_base_dir, 'train', 'labels')
    val_img_dir = os.path.join(output_base_dir, 'val', 'images')
    val_label_dir = os.path.join(output_base_dir, 'val', 'labels')

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Split dataframe into train and validation sets
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    print(f"Total annotations: {len(df)}")
    print(f"Train annotations: {len(train_df)}")
    print(f"Validation annotations: {len(val_df)}")

    # Process training data
    print("Processing training data...")
    for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
        # Changed: Use 'img_id' as per your CSV
        img_id = row['img_id']
        img_full_path = os.path.join(img_dir, img_id)
        
        try:
            img = cv2.imread(img_full_path)
            if img is None:
                print(f"Warning: Could not read image file: {img_full_path}. Skipping.")
                continue
            h, w = img.shape[:2]
        except Exception as e:
            print(f"Error loading image {img_full_path}: {e}. Skipping.")
            continue

        # Convert to YOLO format (normalized cx,cy,width,height)
        x_center = (row['xmin'] + row['xmax']) / (2 * w)
        y_center = (row['ymin'] + row['ymax']) / (2 * h)
        box_w = (row['xmax'] - row['xmin']) / w
        box_h = (row['ymax'] - row['ymin']) / h
        
        # Class 0 for license plate
        yolo_annotation = f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"

        # Get base filename without extension for the label file
        base_filename = os.path.splitext(img_id)[0]
        
        # Write YOLO annotation file
        label_output_path = os.path.join(train_label_dir, f"{base_filename}.txt")
        with open(label_output_path, "w") as f:
            f.write(yolo_annotation)
        
        # Copy image to the new structure
        shutil.copy(img_full_path, os.path.join(train_img_dir, img_id))

    # Process validation data
    print("Processing validation data...")
    for index, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Val"):
        # Changed: Use 'img_id' as per your CSV
        img_id = row['img_id']
        img_full_path = os.path.join(img_dir, img_id)

        try:
            img = cv2.imread(img_full_path)
            if img is None:
                print(f"Warning: Could not read image file: {img_full_path}. Skipping.")
                continue
            h, w = img.shape[:2]
        except Exception as e:
            print(f"Error loading image {img_full_path}: {e}. Skipping.")
            continue

        x_center = (row['xmin'] + row['xmax']) / (2 * w)
        y_center = (row['ymin'] + row['ymax']) / (2 * h)
        box_w = (row['xmax'] - row['xmin']) / w
        box_h = (row['ymax'] - row['ymin']) / h
        
        yolo_annotation = f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"

        base_filename = os.path.splitext(img_id)[0]
        label_output_path = os.path.join(val_label_dir, f"{base_filename}.txt")
        with open(label_output_path, "w") as f:
            f.write(yolo_annotation)
        
        shutil.copy(img_full_path, os.path.join(val_img_dir, img_id))

    print(f"YOLO dataset created at: {output_base_dir}")

if __name__ == "__main__":
    # Define paths relative to the script's location (2_scripts/detection/)
    csv_input_path = "../../1_data/train_detection/labels.csv"
    images_input_dir = "../../1_data/train_detection/images"
    yolo_output_base_dir = "../../1_data/detection_yolo_dataset" # New directory for YOLO format

    # Create dummy data for demonstration if not already present
    if not os.path.exists(csv_input_path):
        print("Creating dummy data for conversion demonstration...")
        os.makedirs(images_input_dir, exist_ok=True)
        dummy_data = {
            'img_id': [f'car_{i:04d}.jpg' for i in range(1, 901)], # Changed to 'img_id' here too for consistency
            'xmin': [50 + i for i in range(900)],
            'ymin': [100 + i // 2 for i in range(900)],
            'xmax': [250 + i for i in range(900)],
            'ymax': [150 + i // 2 + 50 for i in range(900)]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df.to_csv(csv_input_path, index=False)

        for i in range(1, 901):
            img_path = os.path.join(images_input_dir, f'car_{i:04d}.jpg')
            # Create a simple dummy image (e.g., 640x480)
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_img, f"Car {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(img_path, dummy_img)
        print("Dummy data created.")

    create_yolo_dataset(
        csv_path=csv_input_path,
        img_dir=images_input_dir,
        output_base_dir=yolo_output_base_dir
    )
    print("Conversion to YOLO format complete.")