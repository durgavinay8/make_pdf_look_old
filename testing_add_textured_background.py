import cv2
import numpy as np

# Load the scanned black-and-white document
doc_img = cv2.imread(r'C:\Users\LENOVO\Pictures\Screenshots\Screenshot 2025-04-18 002257.png', cv2.IMREAD_GRAYSCALE)

# Invert image to get white text on black (for mask creation)
inverted = 255 - doc_img

# Threshold the inverted image to isolate the text
_, text_mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Convert mask to 3-channel image
text_mask_colored = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)

# Load background using OpenCV
bg_img = cv2.imread(r'C:\Users\LENOVO\Downloads\Phone Link\grunge-paper-background.jpg')  # BGR format
bg_img = cv2.resize(bg_img, (doc_img.shape[1], doc_img.shape[0]))

# Where mask is white (text), place black text; otherwise, keep background
result = np.where(text_mask_colored == 255, (0, 0, 0), bg_img)

# Save the result
cv2.imwrite('output_with_texture.png', result)