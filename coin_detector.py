"""
Euro Coin Detection and Identification System
This is the main orchestrator. It connects the "eyes" (detection) with the "brain" (identification).

Full pipeline orchestrator that combines:
  - coin_detection.py   : locating coins (segmentation, Hough, watershed)
  - coin_identification.py : classifying coins (colour, denomination, value)

How it works:
- Setup: Loads the image from your computer.
- Preprocess: Shrinks the image to a manageable size and analyzes the background.
- Detect: Finds the exact locations and sizes of all coins.
- Analyze Color: Converts the image to HSV and LAB color spaces to understand the metal type.
- Identify: Groups the coins by color (copper, gold, bimetallic) and guesses their exact Euro value.
- Calculate: Adds up the total amount of money found in the image.
- Output: Returns a dictionary with all the math, and can draw the results on the picture.

Usage (in main.py):
    from coin_detector import CoinDetector
    detector = CoinDetector()
    result = detector.process_image('path/to/image.jpg')
"""

import cv2

from coin_detection import CoinDetection
from coin_identification import CoinIdentification


class CoinDetector:
    """End-to-end euro coin detector and identifier."""

    def __init__(self, target_width=800, use_knn=False):
        """
        Initializes the pipeline by setting up the two main worker classes.
        
        Args:
            target_width (int): The baseline width for resizing images. Default is 800px.
            use_knn (bool): If True, uses K-Nearest Neighbors for more accurate 
                            color classification. Default is True.
                            
        Returns:
            None
        """
        # Instantiate the detection module (handles finding the x, y, radius)
        self.detection = CoinDetection(target_width=target_width)

        # Instantiate the identification module (handles color and value assignment)
        # THE KNN METHOD IS NOT USED IN THE FINAL VERSION, BUT IT CAN BE ACTIVATED FOR TESTING PURPOSES.
        self.identification = CoinIdentification(use_knn=use_knn)

    def process_image(self, image_path):
        """
        Runs the full detection and identification pipeline on a single image.
        
        Args:
            image_path (str): The file path to the image you want to process.
            
        Returns:
            dict: A comprehensive results dictionary containing:
                - 'count' (int): Total number of coins detected.
                - 'total_value' (float): Total value of the coins in Euros.
                - 'coins' (list): List of dictionaries, one for each coin, containing location (x, y, r) and properties (color_group, denomination).
                - 'image' (numpy.ndarray): The resized image array used for processing.
                - 'scale' (float): The scaling factor applied to the original image.
                - 'bg_type' (dict): The background analysis results.
                - 'error' (str): Present only if the image failed to load.
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

    # VISUALISATION
    @staticmethod
    def visualize(img, result, title=''):
        """
        Draws the detected coins, their labels, and the total value directly onto the image.
        
        Args:
            img (numpy.ndarray): The image to draw on (usually the image from the result dict).
            result (dict): The dictionary returned by process_image().
            title (str, optional): A title to print at the top of the image.
            
        Returns:
            numpy.ndarray: A copy of the image with all the visual annotations applied.
        """
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
