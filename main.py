import cv2
import numpy as np
from preprocess import preprocess_image
from segment import segment_characters
from recognize import load_templates, recognize_characters

image_path = "input_images/sample 2.png"

binary = preprocess_image(image_path)
chars = segment_characters(binary)

templates = load_templates("templates")
text = recognize_characters(chars, templates)

print("DEBUG OCR TEXT ->", repr(text))

# ---------- OCR RESULT UI ----------
if not text.strip():
    text = "No text detected"

# Create white BGR canvas (IMPORTANT)
canvas = np.ones((200, 600, 3), dtype=np.uint8) * 255

# Title
cv2.putText(
    canvas,
    "OCR RESULT",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    2
)

# OCR Text
cv2.putText(
    canvas,
    text,
    (20, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    2
)

cv2.imshow("OCR Result", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

