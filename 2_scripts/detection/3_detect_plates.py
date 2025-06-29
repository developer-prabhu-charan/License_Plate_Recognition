# 3_detect_plates.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

class YOLODetector:
    def __init__(self, model_path=None):
        """
        Initializes the YOLO detector.
        Args:
            model_path (str, optional): Path to the trained YOLO model weights (e.g., 'path/to/best.pt').
                                        If None, it defaults to the expected path in 3_models.
        """
        if model_path is None:
            # Default path relative to the script's location
            self.model_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '../../3_models/yolo_plate_detector/weights/best.pt'
            ))
        else:
            self.model_path = model_path
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained YOLO model not found at: {self.model_path}. "
                                    f"Please ensure 2_train_yolo.py has been run successfully.")
        
        # Load the YOLO model using ultralytics.YOLO
        self.model = YOLO(self.model_path)
        print(f"YOLO Detector loaded from: {self.model_path}")
    
    def detect(self, img_input):
        """
        Detects license plates in an image.
        Args:
            img_input (str or PIL.Image.Image or np.ndarray): Path to image, PIL Image, or NumPy array (BGR).
        Returns:
            list: A list of dictionaries, each containing 'bbox' (xmin, ymin, xmax, ymax)
                  and 'confidence' for detected license plates.
        """
        # Run inference using the ultralytics.YOLO model object
        results = self.model(img_input, verbose=False) # Suppress verbose output during inference

        plates = []
        for r in results: # 'results' is a list of Results objects (one per input image)
            # Access bounding boxes, confidences, and classes from the Results object
            boxes = r.boxes.xyxy.cpu().numpy()     # Bounding box coordinates (xmin, ymin, xmax, ymax)
            confidences = r.boxes.conf.cpu().numpy() # Confidence scores
            classes = r.boxes.cls.cpu().numpy()   # Class IDs

            for i in range(len(boxes)):
                bbox = boxes[i]
                conf = confidences[i]
                cls = classes[i]

                # Assuming class 0 is 'license_plate'
                if cls == 0:
                    plates.append({
                        'bbox': [int(x) for x in bbox],  # [xmin, ymin, xmax, ymax]
                        'confidence': float(conf)
                    })
        
        return plates

    def crop_plate(self, original_img_np, bbox):
        """
        Crops the license plate region from a NumPy array image given a bounding box.
        
        Args:
            original_img_np (np.ndarray): The original image as a NumPy array (BGR format expected from cv2.imread).
            bbox (list): Bounding box coordinates [xmin, ymin, xmax, ymax].
        
        Returns:
            np.ndarray: The cropped license plate image (NumPy array).
        """
        xmin, ymin, xmax, ymax = bbox
        
        # Ensure coordinates are within image bounds
        h, w = original_img_np.shape[:2]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        # Perform the crop
        cropped_lp = original_img_np[ymin:ymax, xmin:xmax]
        return cropped_lp

# Example usage for testing all images in the test folder:
if __name__ == "__main__":
    detector = YOLODetector() # Loads the default model path

    # Path to your test images directory
    test_images_dir = "../../1_data/test/images/"
    # Output directory for saving detection results
    output_results_dir = "../../1_data/test/detection_results/" 

    # Ensure output directory exists
    os.makedirs(output_results_dir, exist_ok=True)

    # Ensure the test images directory exists
    if not os.path.exists(test_images_dir):
        print(f"Error: Test images directory not found at {test_images_dir}. Please ensure it exists and contains images.")
    else:
        # Get all image files from the test directory
        image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
        image_files.sort() # Sort to process in a consistent order

        if not image_files:
            print(f"No image files found in {test_images_dir}.")
        else:
            print(f"Found {len(image_files)} images in {test_images_dir} for testing.")
            
            for image_filename in image_files:
                img_full_path = os.path.join(test_images_dir, image_filename)
                print(f"\nProcessing image: {image_filename}")

                # Load the image using OpenCV (returns NumPy array in BGR)
                img_np = cv2.imread(img_full_path)

                if img_np is None:
                    print(f"Error: Could not load image from {img_full_path}. Skipping.")
                    continue
                
                plates = detector.detect(img_np) # Pass NumPy array

                if plates:
                    print(f"Detected {len(plates)} license plates in {image_filename}:")
                    display_img = img_np.copy() # Make a copy to draw on for display

                    for i, plate_info in enumerate(plates):
                        bbox = plate_info['bbox']
                        confidence = plate_info['confidence']
                        print(f"  Plate {i+1}: BBox={bbox}, Confidence={confidence:.2f}")

                        # Crop the plate
                        cropped_lp_img = detector.crop_plate(img_np, bbox)

                        # --- Save Visualization ---
                        # Draw bounding box on the original image for visualization
                        xmin, ymin, xmax, ymax = bbox
                        cv2.rectangle(display_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2) # Green rectangle
                        cv2.putText(display_img, f"Conf: {confidence:.2f}", (xmin, ymin - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Save the cropped plate image
                        cropped_output_path = os.path.join(output_results_dir, f"cropped_{os.path.splitext(image_filename)[0]}_{i+1}.jpg")
                        cv2.imwrite(cropped_output_path, cropped_lp_img)
                        print(f"  Cropped plate saved to: {cropped_output_path}")

                    # Save the original image with detections drawn
                    detected_output_path = os.path.join(output_results_dir, f"detected_{image_filename}")
                    cv2.imwrite(detected_output_path, display_img)
                    print(f"  Image with detections saved to: {detected_output_path}")
                else:
                    print(f"No license plates detected in {image_filename}.")
                    # Optionally, save the original image to indicate no detections if needed
                    # no_detection_output_path = os.path.join(output_results_dir, f"no_detection_{image_filename}")
                    # cv2.imwrite(no_detection_output_path, img_np)
                    # print(f"  Original image (no detections) saved to: {no_detection_output_path}")
            print("\nFinished processing all test images. Check the 'detection_results' folder for output.")