import torch
import cv2
import pandas as pd
from PIL import Image
from 2_scripts.detection.3_detect_plates import YOLODetector
from 2_scripts.ocr.2_predict_ocr import OCRPredictor

def main():
    # Load models
    detector = YOLODetector("../../3_models/yolo_plate.pt")
    ocr = OCRPredictor("../../3_models/crnn_ocr.h5")
    
    # Process test images
    submission = pd.read_csv("../../1_data/test/sample_submission.csv")
    for idx, row in submission.iterrows():
        img = Image.open(f"../../1_data/test/images/{row['image_id']}.jpg")
        
        # Detection
        plates = detector.detect(img)
        if plates:
            plate_img = img.crop(plates[0]['bbox'])
            
            # OCR (handles masking)
            plate_text = ocr.predict(plate_img)
            submission.loc[idx, 'plate_number'] = plate_text
    
    submission.to_csv("../../results/final_submission.csv", index=False)

if __name__ == "__main__":
    main()