import cv2
import matplotlib.pyplot as plt

def show_bbox(img_path, bbox):
    """Draw bounding box on image"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    xmin, ymin, xmax, ymax = bbox
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def show_plate_with_text(plate_img, predicted_text):
    """Display plate image with predicted text"""
    plt.figure(figsize=(5,2))
    plt.imshow(plate_img)
    plt.title(f"Predicted: {predicted_text}", fontsize=10)
    plt.axis('off')
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Detection visualization
    show_bbox("../../1_data/test_images/001.jpg", [100, 150, 300, 200])
    
    # OCR visualization
    plate_img = cv2.imread("../../1_data/train_ocr/images/001.jpg")
    show_plate_with_text(plate_img, "ABC123")