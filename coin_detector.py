"""
Euro Coin Detection and Identification System
==============================================
Master 1 Informatique - IFLBX030 Introduction a l'analyse d'images

Full pipeline orchestrator that combines:
  - coin_detection.py   : locating coins (segmentation, Hough, watershed)
  - coin_identification.py : classifying coins (colour, denomination, value)

Usage:
    from coin_detector import CoinDetector
    detector = CoinDetector()
    result   = detector.process_image('path/to/image.jpg')
"""

import cv2

from coin_detection import CoinDetection
from coin_identification import CoinIdentification


class CoinDetector:
    """End-to-end euro coin detector and identifier.

    Parameters
    ----------
    target_width : int
        Images are resized so width ≈ target_width before processing.
    use_knn : bool
        Whether to use KNN for colour classification refinement.
    """

    def __init__(self, target_width=800, use_knn=True):
        self.detection = CoinDetection(target_width=target_width)
        self.identification = CoinIdentification(use_knn=use_knn)

    # ================================================================ #
    #  PIPELINE                                                         #
    # ================================================================ #
    def process_image(self, image_path):
        """Run the full pipeline on a single image.

        Returns a dict with:
            count        – number of coins detected
            total_value  – estimated total value in EUR
            coins        – list of per-coin dicts (x, y, r, denomination, …)
            image        – the preprocessed image (for visualisation)
            scale        – resize scale factor
            bg_type      – background analysis result
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return {'count': 0, 'total_value': 0.0, 'coins': [],
                    'error': str(image_path)}

        # 1. Preprocess
        img_proc, scale = self.detection.preprocess(img)

        # 2. Analyse background
        bg_info = self.detection.analyse_background(img_proc)

        # 3. Detect coins (locations)
        coins = self.detection.detect(img_proc, bg_info)

        # 4. Classify each coin (colour group)
        hsv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img_proc, cv2.COLOR_BGR2Lab)
        for i in range(len(coins)):
            coins[i] = self.identification.classify_one(hsv, lab, coins[i], bg_info)

        # 5. Assign denominations (based on colour + relative size)
        coins = self.identification.assign_denominations(coins)

        # 6. Sum
        total = round(sum(c.get('value', 0) for c in coins), 2)

        return {
            'count': len(coins),
            'total_value': total,
            'coins': coins,
            'image': img_proc,
            'scale': scale,
            'bg_type': bg_info,
        }

    # ================================================================ #
    #  VISUALISATION                                                    #
    # ================================================================ #
    @staticmethod
    def visualize(img, result, title=''):
        """Draw detected coins, labels, and total on the image."""
        vis = img.copy()
        cm = {
            'copper':     (51, 102, 204),
            'gold':       (0, 210, 255),
            'bimetallic': (210, 200, 50),
        }
        for c in result.get('coins', []):
            clr = cm.get(c.get('color_group', ''), (255, 255, 255))
            x, y, r = c['x'], c['y'], c['r']
            cv2.circle(vis, (x, y), r, clr, 2)
            cv2.circle(vis, (x, y), 3, clr, -1)
            lbl = c.get('denomination', '?')
            fs, th = 0.50, 2
            (tw, tht), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            cv2.rectangle(vis, (x-tw//2-3, y-r-tht-10),
                          (x+tw//2+3, y-r-3), (0, 0, 0), -1)
            cv2.putText(vis, lbl, (x-tw//2, y-r-6),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, clr, th)

        n = result.get('count', 0)
        v = result.get('total_value', 0)
        cv2.putText(vis, f"{n} pieces | {v:.2f} EUR", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if title:
            cv2.putText(vis, title, (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return vis
