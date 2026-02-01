import cv2

def preprocess_image(path):
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normal binary threshold (NOT inverted)
    _, thresh = cv2.threshold(
        gray, 180, 255, cv2.THRESH_BINARY
    )

    # Invert AFTER thresholding
    thresh = cv2.bitwise_not(thresh)

    # Dilation to strengthen characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    return thresh
