
import cv2
import numpy as np


def load_image(image_path):
    """
 Image loading
    """
    image = cv2.imread(image_path)

    if image is None:
        raise IOError(f"Erreur : impossible de charger l'image {image_path}")

    return image


def preprocess_image(image):
    """
     Image preprocessing
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def detect_coins(preprocessed_image):
    """
     Coin detection using Hough Circles
    """
    circles = cv2.HoughCircles(
        preprocessed_image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=120
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

    return circles

def extract_coin_rois(image, circles, padding=5):
    """
    Extract coin regions of interest (ROI)
    """
    rois = []

    if circles is None:
        return rois

    h, w = image.shape[:2]

    for (x, y, r) in circles:
        r = r + padding
        x1 = max(x - r, 0)
        y1 = max(y - r, 0)
        x2 = min(x + r, w)
        y2 = min(y + r, h)

        roi = image[y1:y2, x1:x2]

        # apply circular mask
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        center = (roi.shape[1] // 2, roi.shape[0] // 2)
        radius = min(center[0], center[1], r)

        cv2.circle(mask, center, radius, 255, -1)
        roi_masked = cv2.bitwise_and(roi, roi, mask=mask)

        rois.append(roi_masked)

    return rois
