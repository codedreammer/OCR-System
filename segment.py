import cv2

def segment_characters(binary_image):
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 8 and h > 15:
            boxes.append((x, y, w, h))

    # sort left to right
    boxes = sorted(boxes, key=lambda b: b[0])

    characters = []
    for box in boxes:
        x, y, w, h = box
        char_img = binary_image[y:y+h, x:x+w]
        characters.append((x, char_img))

    return characters
