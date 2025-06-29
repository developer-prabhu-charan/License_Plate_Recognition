# 2_train_yolo.py
import os
from ultralytics import YOLO
import yaml

def train_yolo_model(data_yaml_path, epochs=50, batch_size=16, imgsz=640, project_name="yolo_plate_detector"):
    """
    Trains a YOLOv8 model for license plate detection.

    Args:
        data_yaml_path (str): Path to the YAML configuration file for the dataset.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        imgsz (int): Input image size for training.
        project_name (str): Name of the project under which models/results will be saved.
    """
    # Load a pre-trained YOLOv8n model (nano version, good balance of speed/accuracy)
    # You can choose other versions like 'yolov8s.pt' for slightly larger models.
    model = YOLO('yolov8n.pt')

    # Define the output directory for the trained model relative to the project root
    # ultralytics saves under {project}/{name}/weights/best.pt
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../3_models'))

    print(f"Training results will be saved in: {save_dir}")

    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name=project_name, # This will create a subfolder like 'yolo_plate_detector'
        project=save_dir,  # Saves output to 3_models/yolo_plate_detector/
        exist_ok=True,     # Overwrite existing project if it exists
        # Add more arguments as needed: optimizer, lr, patience, etc.
    )

    # After training, the best model is typically saved at
    # {project}/{name}/weights/best.pt
    # E.g., license_plate_project/3_models/yolo_plate_detector/weights/best.pt
    print(f"\nTraining complete. Best model should be saved in: {save_dir}/{project_name}/weights/best.pt")

if __name__ == "__main__":
    # Define paths relative to the script's location (2_scripts/detection/)
    yolo_dataset_base_dir = "../../1_data/detection_yolo_dataset"
    data_yaml_path = "yolo_data.yaml" # This will be created in the same directory as this script

    # Create the data.yaml file dynamically for YOLOv8
    # This path is relative to the directory where the training command is executed by ultralytics
    # which is usually the project root when using model.train() from code.
    # So, the paths inside the YAML should be relative to the 'path' field.
    data_config = {
        'path': os.path.abspath(yolo_dataset_base_dir), # Absolute path to the dataset root
        'train': 'train/images', # Relative to 'path'
        'val': 'val/images',     # Relative to 'path'
        'nc': 1,                 # Number of classes (1 for license plate)
        'names': ['license_plate'] # Class name
    }

    # Write the YAML file
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f, sort_keys=False)
    print(f"YOLO data config file created: {data_yaml_path}")

    # Call the training function
    train_yolo_model(data_yaml_path=data_yaml_path)