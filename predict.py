"""Load the saved models, read plates, and draw a complete plate preview."""
import csv
import hashlib
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
# Set the runtime directory before importing Ultralytics; preserve an existing environment override.
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".runtime", "ultralytics"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Keep the XML dataset label unchanged; it is not a translated UI string.
PLATE_LABEL = "کل ناحیه پلاک"


def file_hash(path):
    # Hash in chunks to keep memory bounded when reading model checkpoints.
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_coordinates(box, width, height):
    # Round starts down and ends up so fractional boundaries do not discard edge pixels.
    # Clamp to image bounds; NumPy crop end coordinates are exclusive.
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("The bounding box does not contain any image pixels.")
    return x1, y1, x2, y2


def minimum_size(image):
    # Enlarge tiny crops proportionally and return actual x/y scales for coordinate recovery.
    height, width = image.shape[:2]
    scale = max(1, 10 / min(height, width))
    new_width, new_height = int(np.ceil(width * scale)), int(np.ceil(height * scale))
    resized = cv2.resize(image, (new_width, new_height)) if scale > 1 else image
    return resized, new_width / width, new_height / height


def character_symbol(label):
    # Map dataset class names to display symbols without changing the trained class IDs.
    return {"الف": "ا", "ه\u200d": "ه", "ژ (معلولین و جانبازان)": "♿"}.get(label, label)


def character_features(image):
    """The SVM sees Otsu + 32x32 padding + 324 HOG features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Otsu chooses the threshold; inversion makes dark characters white foreground.
    height, width = binary.shape
    scale = min(26 / width, 26 / height)
    resized = cv2.resize(binary, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((32, 32), np.uint8)
    # Center the character with a margin while preserving its aspect ratio.
    y, x = (32 - resized.shape[0]) // 2, (32 - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
    # Nine blocks, four cells per block, and nine orientation bins produce 324 features.
    return hog.compute(canvas).reshape(-1).astype(np.float32)


def load_models(folder=None):
    # Load checkpoint names, confidence thresholds, and input sizes from saved settings.
    folder = folder or os.path.join(ROOT, "models")
    with open(os.path.join(folder, "settings.csv"), encoding="utf-8-sig", newline="") as stream:
        settings = {row["model"]: row for row in csv.DictReader(stream)}
    models = []
    for name in ("plate", "character"):
        path = os.path.join(folder, settings[name]["filename"])
        # Verify each checkpoint hash before loading so weights cannot silently mismatch settings.
        if file_hash(path) != settings[name]["sha256"]:
            raise ValueError(f"Model does not match its saved settings: {path}")
        models.append(YOLO(path))
    return models[0], models[1], settings


def rectify_crop(image):
    """Optional border heuristic. Return None when the border is uncertain."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if min(gray.shape) < 10:
        return None
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    # Close edge gaps to make a four-sided plate border easier to recover.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_image, best_score, best_angle = None, -1, 0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        # Try two polygon tolerances on the largest contours to find plausible four-corner borders.
        for fraction in (0.02, 0.04):
            quad = cv2.approxPolyDP(contour, fraction * cv2.arcLength(contour, True), True)
            area = abs(cv2.contourArea(quad))
            if len(quad) != 4 or not cv2.isContourConvex(quad) or area < gray.size * 0.12:
                continue
            points = quad.reshape(4, 2).astype(np.float32)
            sums, differences = points.sum(axis=1), points[:, 0] - points[:, 1]
            # Order corners as top-left, top-right, bottom-right, bottom-left.
            corners = points[[sums.argmin(), differences.argmax(), sums.argmax(), differences.argmin()]]
            if len(np.unique(corners, axis=0)) != 4:
                continue
            top, right, bottom, left = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
            if min(left, right) < 10 or not 2 <= (top + bottom) / (left + right) <= 7:
                continue
            width, height = round(max(top, bottom)), round(max(left, right))
            scale = min(1, 1024 / width)
            width, height = max(2, round(width * scale)), max(2, round(height * scale))
            target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
            transform = cv2.getPerspectiveTransform(corners, target)
            # Map the four detected corners to a rectangle to correct perspective distortion.
            crop = cv2.warpPerspective(image, transform, (width, height))
            bright_fraction = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) >= 110).mean()
            # Brightness coverage and relative contour area are heuristic candidate scores.
            if bright_fraction < 0.45:
                continue
            score = 0.6 * bright_fraction + 0.4 * area / gray.size
            if score > best_score:
                best_image, best_score = crop, score
                delta = corners[1] - corners[0]
                best_angle = np.degrees(np.arctan2(delta[1], delta[0]))
    # Apply correction only when the selected candidate is tilted by at least 12 degrees.
    return best_image if abs(best_angle) >= 12 else None


def read_plates(image, plate_model, character_model, settings, device="cpu", rectify=False):
    # Input is BGR: first detect full-image plates, then detect characters inside each crop.
    if image is None or image.size == 0:
        raise ValueError("The input image could not be read.")
    # The saved plate confidence is a detection threshold, not a probability that the whole text is correct.
    result = plate_model.predict(image, conf=float(settings["plate"]["confidence"]),
                                 imgsz=int(settings["plate"]["image_size"]), device=device, iou=0.7, verbose=False)[0]
    plates = []
    for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
        x1, y1, x2, y2 = crop_coordinates(box, image.shape[1], image.shape[0])
        crop, corrected = image[y1:y2, x1:x2], False
        if rectify:
            # Add a 6% margin to improve the chance of retaining the border for rectification.
            pad_x, pad_y = (x2-x1) * 0.06, (y2-y1) * 0.06
            a, b, c, d = crop_coordinates([x1-pad_x, y1-pad_y, x2+pad_x, y2+pad_y], image.shape[1], image.shape[0])
            candidate = rectify_crop(image[b:d, a:c])
            if candidate is not None:
                crop, corrected = candidate, True
        model_input, scale_x, scale_y = minimum_size(crop)
        # Class-agnostic NMS also suppresses overlapping boxes assigned to different character classes.
        result_chars = character_model.predict(model_input, conf=float(settings["character"]["confidence"]),
                                               imgsz=int(settings["character"]["image_size"]), device=device,
                                               iou=0.7, agnostic_nms=True, verbose=False)[0]
        characters = []
        # Map detections back to crop coordinates; after rectification, these refer to the corrected crop.
        for cb, label, score in zip(result_chars.boxes.xyxy.cpu().tolist(), result_chars.boxes.cls.cpu().tolist(), result_chars.boxes.conf.cpu().tolist()):
            characters.append({"label": character_model.names[int(label)], "confidence": score,
                               "box_in_crop": [cb[0]/scale_x, cb[1]/scale_y, cb[2]/scale_x, cb[3]/scale_y]})
        # Build reading order from character centers, physically left to right.
        characters.sort(key=lambda char: (char["box_in_crop"][0] + char["box_in_crop"][2]) / 2)
        text = "".join(character_symbol(char["label"]) for char in characters)
        # Plate boxes use source-image coordinates; character boxes use local crop coordinates.
        plates.append({"box": box, "confidence": confidence, "text": text, "crop": crop,
                       "characters": characters, "perspective_applied": corrected})
    return plates


def straighten_preview(plate):
    """Small display-only rotation; recognition results are not changed."""
    image, characters = plate["crop"], plate["characters"]
    # Do not rotate a rectified crop again or estimate tilt from fewer than three characters.
    if plate["perspective_applied"] or len(characters) < 3:
        return image, characters
    centers = np.float32([[(c["box_in_crop"][0]+c["box_in_crop"][2])/2,
                           (c["box_in_crop"][1]+c["box_in_crop"][3])/2] for c in characters])
    # Estimate display tilt from a robust line fitted through character centers.
    vx, vy, _, _ = cv2.fitLine(centers, cv2.DIST_L1, 0, 0.01, 0.01).flatten()
    angle = float(np.degrees(np.arctan2(vy, vx)))
    angle = (angle + 90) % 180 - 90
    if abs(angle) > 25:
        return image, characters
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    # Expand the canvas and recenter the transform to avoid clipping rotated image corners.
    new_width = int(np.ceil(width*abs(matrix[0, 0]) + height*abs(matrix[0, 1])))
    new_height = int(np.ceil(height*abs(matrix[0, 0]) + width*abs(matrix[0, 1])))
    matrix[:, 2] += [(new_width-width)/2, (new_height-height)/2]
    rotated = cv2.warpAffine(image, matrix, (new_width, new_height), borderValue=(238, 242, 246))
    mapped = []
    for char in characters:
        x1, y1, x2, y2 = char["box_in_crop"]
        # Transform all four corners into enclosing boxes; leave recognition data unchanged.
        points = np.float32([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]]) @ matrix.T
        mapped.append({**char, "box_in_crop": [points[:, 0].min(), points[:, 1].min(),
                                               points[:, 0].max(), points[:, 1].max()]})
    return rotated, mapped


def draw_plates(plates):
    """Complete plate images with character rectangles and aligned symbols below."""
    # Use a font with Persian glyphs for individual character labels.
    font = ImageFont.truetype("arial.ttf" if os.name == "nt" else "DejaVuSans.ttf", 32)
    panels = []
    for index, plate in enumerate(plates, 1):
        image, characters = straighten_preview(plate)
        # OpenCV provides BGR arrays; Pillow expects RGB images.
        crop = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        scale = 900 / crop.width
        # Scale the image and character coordinates together to a fixed preview width.
        crop = crop.resize((900, max(1, round(crop.height * scale))))
        panel = Image.new("RGB", (948, crop.height + 100), "#eef2f6")
        panel.paste(crop, (24, 36))
        draw = ImageDraw.Draw(panel)
        draw.text((24, 8), f"Plate #{index} - score {plate['confidence']:.2f}", fill="black", font=ImageFont.load_default(size=18))
        for char in characters:
            x1, y1, x2, y2 = char["box_in_crop"]
            draw.rectangle((24+x1*scale, 36+y1*scale, 24+x2*scale, 36+y2*scale), outline="#00a66c", width=3)
            center = 24 + (x1+x2) * scale / 2
            draw.line((center, 36+y2*scale, center, crop.height+44), fill="#00a66c")
            draw.text((center, crop.height+68), character_symbol(char["label"]), anchor="mm", fill="#12375a", font=font)
        if not characters:
            draw.text((24, crop.height+55), "No characters detected.", fill="black", font=ImageFont.load_default(size=20))
        panels.append(panel)
    if not panels:
        result = Image.new("RGB", (600, 100), "white")
        ImageDraw.Draw(result).text((20, 35), "No plate detected.", fill="black", font=ImageFont.load_default(size=24))
        return result
    # Stack every detected plate panel vertically into one output image.
    result = Image.new("RGB", (948, sum(panel.height for panel in panels)), "#eef2f6")
    y = 0
    for panel in panels:
        result.paste(panel, (0, y))
        y += panel.height
    return result


def save_predictions(plates, folder):
    # Save a visual preview and machine-readable CSV; the BOM helps Excel read Persian text.
    os.makedirs(folder, exist_ok=True)
    draw_plates(plates).save(os.path.join(folder, "prediction.png"))
    with open(os.path.join(folder, "prediction.csv"), "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["plate", "text", "confidence", "x1", "y1", "x2", "y2", "perspective_applied"])
        for index, plate in enumerate(plates, 1):
            writer.writerow([index, plate["text"], plate["confidence"], *plate["box"], plate["perspective_applied"]])
