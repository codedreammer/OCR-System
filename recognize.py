import cv2
import os
import numpy as np


# ---------- HELPER FUNCTIONS ----------

def crop_char_from_image(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]


def resize_with_padding(img, size=30):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.uint8)

    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    canvas = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


# ---------- TEMPLATE LOADING ----------

def load_templates(folder="templates"):
    templates = {}

    if not os.path.exists(folder):
        print(f"Template folder not found: {folder}")
        return templates

    valid_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for file in valid_files:
        name = os.path.splitext(file)[0]
        char = None

        if name.startswith("LC_"):
            char = name.replace("LC_", "")
        elif name.startswith("UC_"):
            char = name.replace("UC_", "")
        elif name.startswith("D_"):
            char = name.replace("D_", "")

        if char is None:
            continue

        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.dilate(img, kernel, iterations=1)

        img = crop_char_from_image(img)
        img = resize_with_padding(img, 30)

        templates[char] = img

    print(f"Loaded {len(templates)} templates.")
    return templates


# ---------- LEVEL 2 GEOMETRIC FILTER ----------

def filter_templates_by_geometry(char_img, templates):
    """
    Reduce confusion between geometrically similar characters
    using aspect-ratio heuristics.
    """
    h, w = char_img.shape
    aspect_ratio = h / (w + 1e-5)

    filtered = {}

    for char, tmpl in templates.items():
        # Uppercase letters are generally taller
        if aspect_ratio > 1.6:
            if char.isupper():
                filtered[char] = tmpl
        # Lowercase letters are generally wider / shorter
        elif aspect_ratio < 1.2:
            if char.islower():
                filtered[char] = tmpl
        else:
            filtered[char] = tmpl

    return filtered if filtered else templates


# ---------- CHARACTER RECOGNITION ----------

def recognize_characters(characters, templates):
    result = ""
    prev_x = None
    prev_w = None

    for x, char_img in characters:
        char_img = resize_with_padding(char_img, 30)

        scores = []

        for char, tmpl in templates.items():
            res = cv2.matchTemplate(
                char_img, tmpl, cv2.TM_CCOEFF_NORMED
            )
            score = res[0][0]
            scores.append((score, char))

        # Sort by score (best first)
        scores.sort(reverse=True)

        best_score, best_match = scores[0]

        # Relative spacing
        h, w = char_img.shape
        if prev_x is not None and prev_w is not None:
            if (x - prev_x) > (prev_w * 1.3):
                result += " "

        # Accept character
        if best_score > 0.35:
            result += best_match
            prev_x = x
            prev_w = w

    return result
