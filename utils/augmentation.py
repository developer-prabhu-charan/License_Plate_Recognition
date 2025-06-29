import cv2
import numpy as np
import random

def apply_mask(img):
    """Simulate random masking on license plates"""
    h, w = img.shape[:2]
    
    # Random rectangular mask
    mask_w = random.randint(10, int(w*0.4))  # Mask up to 40% width
    mask_h = random.randint(5, int(h*0.3))   # Mask up to 30% height
    x = random.randint(0, w - mask_w)
    y = random.randint(0, h - mask_h)
    
    img[y:y+mask_h, x:x+mask_w] = 0  # Black mask
    return img

def augment_plate(img_path):
    """Full augmentation pipeline"""
    img = cv2.imread(img_path)
    
    # Random transformations
    if random.random() > 0.5:
        img = apply_mask(img)
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (3,3), 0)
    if random.random() > 0.5:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    
    return img

# Example usage:
if __name__ == "__main__":
    augmented = augment_plate("../../1_data/train_ocr/images/001.jpg")
    cv2.imwrite("augmented_plate.jpg", augmented)