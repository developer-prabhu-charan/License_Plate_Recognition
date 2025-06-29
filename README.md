# License Plate Recognition Project

This project implements an end-to-end pipeline for recognizing license plates from images. It involves two main stages: license plate detection using a YOLO model and character recognition using an OCR model.

## Directory Structure

*   **`1_data/`**: Contains all the datasets used for training and testing the models.
    *   `detection_yolo_dataset/`: Data for training the YOLOv8 license plate detector.
    *   `recognition_ocr_dataset/`: Data for training the OCR character recognizer.
    *   `test/`: Test images and sample submission file.
    *   `train_detection/`: Original training data for detection (likely pre-conversion to YOLO format).
    *   `train_ocr/`: Original training data for OCR.
*   **`2_scripts/`**: Contains Python scripts for the different parts of the pipeline.
    *   `detection/`: Scripts related to license plate detection (data conversion, training, inference).
    *   `ocr/`: Scripts related to character recognition (data preparation, training, inference).
    *   `pipeline/`: Script to run the complete end-to-end license plate recognition process.
*   **`3_models/`**: Stores the trained models.
    *   `ocr_recognizer/`: The trained OCR model.
    *   `yolo_plate_detector/`: The trained YOLOv8 plate detection model.
*   **`5_output/`**: Directory where the output of the recognition pipeline is saved.
    *   `recognized_plates/`: Contains images with detected and recognized plates, and a CSV file with the results.
*   **`utils/`**: Utility scripts for various tasks like data augmentation, data handling, and visualization.

## Pipeline Overview

The license plate recognition process consists of the following steps:

1.  **License Plate Detection**: A YOLOv8 model is used to detect the location of license plates in an input image.
2.  **Character Recognition (OCR)**: Once a license plate is detected, the plate region is cropped and fed into an OCR model to recognize the characters on the plate.

## Running the Project

### Dependencies

This project relies on several Python libraries. You can typically install them using pip:

```bash
pip install pandas opencv-python scikit-learn tqdm numpy ultralytics torch torchvision Pillow PyYAML
```

It's recommended to use a virtual environment to manage dependencies.

*   **Python 3.x**
*   **pandas**: For data manipulation, especially CSV files.
*   **OpenCV (cv2)**: For image processing tasks.
*   **scikit-learn**: Used for splitting data (e.g., train/test split).
*   **tqdm**: For displaying progress bars.
*   **NumPy**: For numerical operations, especially with image data.
*   **Ultralytics YOLO**: The framework used for the YOLOv8 object detection model.
*   **PyTorch**: The deep learning framework used for training the OCR model (and by YOLO).
*   **torchvision**: Part of PyTorch, provides image transformation utilities.
*   **Pillow (PIL)**: For image manipulation.
*   **PyYAML**: For handling YAML configuration files (used by YOLO).

### Training the Models

1.  **Train YOLO Plate Detector**:
    *   Convert your dataset to YOLO format using `2_scripts/detection/1_convert_to_yolo.py`.
    *   Train the YOLO model using `2_scripts/detection/2_train_yolo.py`.
2.  **Train OCR Model**:
    *   Prepare the OCR training data using `2_scripts/ocr/1_prepare_ocr_data.py`.
    *   Train the OCR model using `2_scripts/ocr/2_train_ocr.py`.

### Running the End-to-End Pipeline

To run the full license plate recognition pipeline on new images:

1.  Ensure your trained models are in the `3_models/` directory.
2.  Place the input images in a suitable directory.
3.  Run the pipeline script: `python 2_scripts/pipeline/run_end_to_end.py`
    *   You might need to modify the script to point to your input image directory and output directory.

## Datasets

The project uses custom datasets for training the detection and OCR models, located in the `1_data/` directory. The detection dataset is expected to be in YOLO format after conversion, and the OCR dataset consists of cropped license plate images and their corresponding character sequences.

## Models

*   **YOLOv8 Plate Detector**: A YOLOv8 model fine-tuned for detecting license plates.
*   **OCR Recognizer**: A model (architecture details can be added here if known) trained to recognize characters from license plate images.

---

This `README.md` provides a good starting point. It can be further improved by adding more specific details about dependencies, model architectures, and more detailed usage instructions.
