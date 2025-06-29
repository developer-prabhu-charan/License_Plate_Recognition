import cv2
import pandas as pd
import numpy as np

def load_ocr_data():
    df = pd.read_csv("../../1_data/train_ocr/labels.csv")
    char_to_num = {c:i for i,c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    
    X, y = [], []
    for _, row in df.iterrows():
        img = cv2.imread(f"../../1_data/train_ocr/images/{row['image_id']}.jpg")
        img = cv2.resize(img, (200, 50))  # CRNN input size
        X.append(img)
        y.append([char_to_num[c] for c in row['plate_number']])
    
    return np.array(X), np.array(y)