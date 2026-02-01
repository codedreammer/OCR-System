import cv2
import os
import numpy as np

def crop_char_from_image(img):
    """
    Crops the character from the image by finding the largest contour.
    This ensures we only match the character pixels, not empty space.
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]

def resize_with_padding(img, size=30):
    """
    Resizes image to (size, size) while maintaining aspect ratio.
    Pads with black pixels.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=np.uint8)
        
    scale = size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    canvas = np.zeros((size, size), dtype=np.uint8)
    # Center the image
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def load_templates(folder="templates"):
    templates = {}
    
    # Ensure folder exists
    if not os.path.exists(folder):
        print(f"Template folder not found: {folder}")
        return templates

    valid_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for file in valid_files:
        name = os.path.splitext(file)[0]
        char = None

        if name.startswith("LC_"):
            char = name.replace("LC_", "")
        elif name.startswith("D_"):
            char = name.replace("D_", "")
        elif name.startswith("UC_"):
            char = name.replace("UC_", "")
        
        if char is None:
            continue

        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 1. Threshold to ensure Binary (White text on Black BG)
        # Templates are typically Black text on White BG, so THRESH_BINARY_INV gives White on Black
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

        # 2. MATCH PREPROCESSING: Add Dilation
        # The input characters (from preprocess.py) are dilated. 
        # Templates must be dilated too to match the stroke thickness.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.dilate(img, kernel, iterations=1)

        # 3. Crop tight (to match segment.py output which is a bounding box crop)
        img = crop_char_from_image(img)

        # 4. Resize with Padding (Preserve Aspect Ratio)
        # Pure resize was distorting shapes (e.g. 'I' becoming a square).
        img = resize_with_padding(img, 30)

        templates[char] = img

    print(f"Loaded {len(templates)} templates.")
    return templates

def recognize_characters(characters, templates):
    result = ""
    prev_x = None

    for x, char_img in characters:
        # Resize input with padding (matching template logic)
        char_img = resize_with_padding(char_img, 30)

        best_match = None
        best_score = -1

        for char, tmpl in templates.items():
            res = cv2.matchTemplate(
                char_img, tmpl, cv2.TM_CCOEFF_NORMED
            )
            score = res[0][0]

            if score > best_score:
                best_score = score
                best_match = char

        # Threshold
        # With correct alignment/sizing, scores should be decent (>0.5).
        # Relaxed to 0.4 to be safe against noise, but much higher than 0.08.
        if best_score > 0.4:
            # Space detection logic (simple x-gap)
            if prev_x is not None and x - prev_x > 40:
                result += " "
            
            result += best_match
            prev_x = x
    
    return result
