import cv2
import numpy as np
file_path = "D:\\NaanMudhalvan\\Project\\priya\\mri.jpg"
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Failed to load image. Check the file path or format.")
    exit()
img_resized = cv2.resize(img, (512, 512))
img_equalized = cv2.equalizeHist(img_resized)
_, thresh = cv2.threshold(img_equalized, 100, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_output = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500: 
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_output, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Equalized", img_equalized)
cv2.imshow("Threshold", thresh)
cv2.imshow("Highlighted Regions", img_output)
cv2.waitKey(0)
cv2.destroyAllWindows()

