"""
Coin Detection Module
=====================
Locates coins in an image using a combination of:
  - Adaptive segmentation (Otsu on grayscale or saturation channel)
  - Watershed algorithm for touching-coin separation
  - Hough Circle Transform (multi-parameter)
  - Contour analysis with circularity filtering

All methods are merged via a "best-of" strategy with non-maximum suppression
and radius consistency filtering.
"""

import cv2
import numpy as np

# Constant representing the real-world ratio between the smallest (1c) 
# and the largest (2€) Euro coins (approx 16.25mm / 25.75mm)
MIN_RADIUS_RATIO = 0.52

class CoinDetection:
    """Detect coin locations (x, y, r) in a preprocessed image."""

    #################
    # INITIALIZATION
    #################
    def __init__(self, target_width=800):
        """Initialize detection parameters and validation cache."""
        self.target_width = target_width
        # Cache to store image properties for the fast validation step
        self._val_gray = None
        self._val_h = self._val_w = 0
        self._val_bg_gray = 0.0
        self._val_is_coloured = False
        self._val_sat = None
        self._val_bg_sat = 0.0

    ###############
    # PREPROCESSING
    ###############
    def preprocess(self, img):
        """Resizes the image to a standard width to ensure consistent detection parameters."""
        h, w = img.shape[:2]
        scale = self.target_width / w
        new_w = self.target_width
        new_h = int(h * scale)
        
        # Limit height to 1000px to maintain processing speed on very long images
        if new_h > 1000:
            scale = 1000.0 / h
            new_h = 1000
            new_w = int(w * scale)
            
        # INTER_AREA is preferred for shrinking images to avoid aliasing
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

    def analyse_background(self, img):
        """Analyzes image borders to detect if the background is light/dark or highly saturated."""
        h, w = img.shape[:2]
        # Define a border thickness (approx 4% of dimensions)
        b = max(int(min(h, w) * 0.04), 5)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Gather indices for the top, bottom, left, and right border pixels
        idx = np.concatenate([
            np.ravel_multi_index(np.mgrid[:b, :w], (h, w)).ravel(),
            np.ravel_multi_index(np.mgrid[h-b:h, :w], (h, w)).ravel(),
            np.ravel_multi_index(np.mgrid[:h, :b], (h, w)).ravel(),
            np.ravel_multi_index(np.mgrid[:h, w-b:w], (h, w)).ravel(),
        ])
        
        flat = hsv.reshape(-1, 3)
        border = flat[idx]
        # Use median to avoid influence of outliers (like a coin touching the border)
        med_h, med_s, med_v = np.median(border, axis=0)
        
        return {
            'is_light': med_v > 127, 
            'is_coloured': med_s > 115, # Sensitivity threshold for colored backgrounds
            'hue': med_h, 'sat': med_s, 'val': med_v,
        }

    ##############
    # SEGMENTATION
    ##############
    def _segment(self, img, bg_info):
        """Directs the image to the appropriate segmentation method based on background color."""
        if bg_info['is_coloured']:
            return self._segment_coloured(img, bg_info)
        return self._segment_neutral(img, bg_info['is_light'])

    def _segment_neutral(self, img, is_light):
        """Binary segmentation for white/gray/black backgrounds using grayscale intensity."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply CLAHE to improve local contrast (helps with shadows)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Strong blur to smooth out internal coin engravings (stars, faces, numbers)
        blurred = cv2.GaussianBlur(enhanced, (13, 13), 3)
        
        # IMPORTANT: Adaptive Threshold instead of simple Otsu
        # This allows detecting coins even in shadow or on textured wood
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV if is_light else cv2.THRESH_BINARY,
                                     21, 4) # BlockSize 21, C=4
        
        # Morphological operations to clean up the mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # CLOSE fills internal holes (reflections); OPEN removes small external noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=4)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
        
        # Final clean-up: force borders to zero
        # bdr = 6
        # binary[:bdr, :] = 0; binary[-bdr:, :] = 0; binary[:, :bdr] = 0; binary[:, -bdr:] = 0
        return binary

    def _segment_coloured(self, img, bg_info):
        """Segmentation using the Saturation channel for colored surfaces."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        blurred_s = cv2.GaussianBlur(sat, (13, 13), 3)
        
        # Low saturation objects (metal coins) on high saturation backgrounds (colored paper)
        _, binary = cv2.threshold(blurred_s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=4)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
        return binary

    ############
    # VALIDATION
    ############
    def _precompute_validation(self, img, bg_info):
        """Caches background statistics to verify coin candidates later."""
        h, w = img.shape[:2]
        self._val_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._val_h, self._val_w = h, w
        
        bdr = max(int(min(h, w) * 0.04), 5)
        bm = np.zeros((h, w), dtype=np.uint8)
        bm[:bdr, :] = 255; bm[-bdr:, :] = 255; bm[:, :bdr] = 255; bm[:, -bdr:] = 255
        
        # Calculate mean gray intensity of the borders
        self._val_bg_gray = cv2.mean(self._val_gray, mask=bm)[0]
        self._val_is_coloured = bg_info.get('is_coloured', False)
        
        if self._val_is_coloured:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._val_sat = hsv[:, :, 1]
            # Calculate mean saturation of the borders
            self._val_bg_sat = cv2.mean(self._val_sat, mask=bm)[0]
            
        # NEW: Compute a map of sharp edges (Canny)
        # This allows validating a coin even if it has the same color as the background
        # Thresholds 50/150 are standard for detecting metallic edges
        self._val_edges = cv2.Canny(self._val_gray, 50, 150)

    def _validate_fast(self, x, y, r):
        """Validates if candidate is a coin via color difference OR presence of edge."""
        h, w = self._val_h, self._val_w
        if x - r < 4 or y - r < 4 or x + r > w - 4 or y + r > h - 4:
            return False

        # 1. COLOR CRITERIA (The "Donut Test")
        # Compare inside (coin) vs immediate outside (background)
        mask_in = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_in, (x, y), int(r * 0.7), 255, -1)
        
        mask_out = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_out, (x, y), int(r * 1.2), 255, -1)
        cv2.circle(mask_out, (x, y), r, 0, -1) # Remove inside to get the ring

        mean_in = cv2.mean(self._val_gray, mask=mask_in)[0]
        mean_out = cv2.mean(self._val_gray, mask=mask_out)[0]
        diff_color = abs(mean_in - mean_out)

        # 2. EDGE CRITERIA (New)
        # Check if a sharp edge (Canny) exists under the circle perimeter
        mask_edge = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_edge, (x, y), r, 255, 2) # Just the contour (thickness 2)
        
        # Average of white Canny pixels under the circle
        # The higher, the more the circle follows a real edge
        edge_score = cv2.mean(self._val_edges, mask=mask_edge)[0]

        # --- DECISION ---
        
        # Case A: Strong color contrast (e.g. Gold coin on Gray background)
        # Validate directly, even if the edge is blurry.
        if diff_color > 15: 
            return True
            
        # Case B: Camouflage (e.g. Gray coin on Gray background)
        # Color is same (diff < 15), BUT there is a sharp edge (high edge_score).
        # Threshold 30 corresponds to approx 12% of perimeter detected as "strong edge".
        if edge_score > 30: 
            return True

        # Otherwise, it's probably noise (flower, spot)
        return False

    #################
    # HOUGH TRANSFORM
    #################
    def _detect_hough(self, img, gray_blur=None):
        """Finds geometric circles using the Hough Gradient method with a parameter sweep."""
        if gray_blur is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # (15, 15) blur specifically for Hough to ignore internal details
            gray_blur = cv2.GaussianBlur(clahe.apply(gray), (15, 15), 3)
            
        h, w = img.shape[:2]
        min_dim = min(h, w)
        min_r, max_r = max(int(min_dim * 0.04), 22), int(min_dim * 0.28)
        
        # Min distance between centers prevents detecting internal patterns as separate coins
        min_dist = max(int(min_r * 2.1), 55)

        # Iterate through different sensitivities to find the best match
        # param2 is the voting threshold (lower = more false circles)
        for p1, p2 in [(120, 55), (100, 45), (85, 35)]:
            circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=min_dist, param1=p1, param2=p2,
                minRadius=min_r, maxRadius=max_r)
            
            if circles is not None:
                out = []
                for c in np.round(circles[0]).astype(int):
                    if self._validate_fast(c[0], c[1], c[2]):
                        out.append({'x': int(c[0]), 'y': int(c[1]), 'r': int(c[2])})
                # Apply NMS with a high ratio to prevent overlapping detections
                if out: return self._nms(out, ratio=0.85)
        return []

    ###########
    # WATERSHED
    ###########
    def _detect_watershed(self, img, binary):
        """Separates touching coins using distance transform and the Watershed algorithm."""
        if binary is None or np.sum(binary) == 0: return []
        
        # Calculate distance to background for each foreground pixel
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # High threshold to find the 'certain' centers of the coins
        thresh = max(0.28, 0.42 * dist_norm.max()) 
        _, sure_fg = cv2.threshold(dist_norm, thresh, 1.0, 0)
        sure_fg = np.uint8(sure_fg * 255)
        
        # Define 'unknown' regions to be flooded by Watershed
        unknown = cv2.subtract(cv2.dilate(binary, None, iterations=3), sure_fg)
        
        # Label the seeds (markers)
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1; markers[unknown == 255] = 0
        
        img_3ch = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_3ch, markers)
        
        coins = []
        for label in range(2, num_labels + 1):
            mask = np.uint8(markers == label) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: continue
            cnt = max(cnts, key=cv2.contourArea)
            # Area filter to exclude noise
            if cv2.contourArea(cnt) < (np.pi * 18**2): continue
            
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if self._validate_fast(int(cx), int(cy), int(radius)):
                coins.append({'x': int(cx), 'y': int(cy), 'r': int(radius)})
        return coins

    ####################
    # CONTOUR DETECTION
    ####################
    def _detect_contour_simple(self, img, binary):
        """Detects coins based on the shapes (contours) found in the binary mask."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coins = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)
            if perim == 0: continue
            
            # Circularity metric: 4*pi*Area / Perimeter^2 (1.0 = perfect circle)
            circ = 4 * np.pi * area / (perim ** 2)
            if circ < 0.72: continue # Strict circularity check
            
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            # Solidity check: ensures the coin isn't just a crescent or ring shape
            if area / (np.pi * r**2) < 0.75: continue
            
            if self._validate_fast(int(cx), int(cy), int(r)):
                coins.append({'x': int(cx), 'y': int(cy), 'r': int(r)})
        return coins

    #################
    # POST-PROCESSING
    ###############
    @staticmethod
    def _nms(coins, ratio=0.85):
        """Non-Maximum Suppression: removes overlapping circles by keeping the largest ones."""
        if len(coins) <= 1: return coins
        
        # Sort by radius (descending) to prioritize larger, more likely detections
        coins = sorted(coins, key=lambda c: c['r'], reverse=True)
        keep = []
        for c in coins:
            # Check if current coin center is significantly far from already accepted coins
            if not any(np.hypot(c['x']-k['x'], c['y']-k['y']) < (c['r']+k['r']) * 0.80 for k in keep):
                keep.append(c)
        return keep

    @staticmethod
    def _filter_radii(coins):
        """Removes outlier circles whose radii do not match the expected Euro coin proportions."""
        if len(coins) <= 2: return coins
        radii = [c['r'] for c in coins]
        med = float(np.median(radii))
        lo, hi = med * MIN_RADIUS_RATIO, med / MIN_RADIUS_RATIO
        return [c for c in coins if lo <= c['r'] <= hi]

    ##################
    # DETECTION PIPELINE
    ##################
    def detect(self, img, bg_info):
        """Runs the complete detection pipeline and merges results from different strategies."""
        self._precompute_validation(img, bg_info)
        binary = self._segment(img, bg_info)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_blur = cv2.GaussianBlur(clahe.apply(self._val_gray), (15, 15), 3)

        # Run multiple strategies to capture different coin conditions
        coins_ws = self._detect_watershed(img, binary)
        coins_cs = self._detect_contour_simple(img, binary)
        coins_h  = self._detect_hough(img, gray_blur)

        # Merge results: Start with Hough (geometrically the most precise)
        merged = list(coins_h)
        for other in [coins_ws, coins_cs]:
            for oc in other:
                # Add detection only if it doesn't significantly overlap a piece already found
                # Inclusion check: prevents counting internal stars/features
                if not any(np.hypot(oc['x']-m['x'], oc['y']-m['y']) < max(oc['r'], m['r']) * 0.85 for m in merged):
                    merged.append(oc)

        # Final cleaning pass
        merged = self._nms(merged, ratio=0.85)
        return self._filter_radii(merged)