"""
Coin Detection Module
This module finds Euro coins in an image, even if they are touching or placed on difficult backgrounds.

- Preprocessing: Resizes the image to a standard size and checks if the background is colored or gray.
- Segmentation: Creates a black-and-white mask of the coins. It uses color (Saturation) for bright backgrounds, or brightness (Grayscale + Adaptive Threshold) for normal ones.
- Detection: Finds coin shapes using 3 different methods to be sure not to miss any:
  - Hough Transform (looks for perfect circles).
  - Watershed (cuts apart coins that are touching each other).
  - Contours (looks for round, solid shapes in the mask).
- Validation: Acts as a "bouncer". It checks the real image to see if the detected circle actually looks like a coin (checking for shadows, edges, and color differences).
- Post-Processing: Merges the results from the 3 methods, removes duplicates (overlapping circles), and deletes sizes that are impossible for real Euro coins.
"""

import cv2
import numpy as np

# Real-world ratio between the smallest (1c) and largest (2€) Euro coins (approx 16.25mm / 25.75mm). We use 0.52 to leave a bit of margin for perspective distortion or slight angle shifts in the image.
MIN_RADIUS_RATIO = 0.52

class CoinDetection:
    """Detects coin locations (x, y, r) in a given image."""

    # INITIALIZATION
    def __init__(self, target_width=800):
        """
        Initializes the detector with a standard processing width and sets up a cache.
        
        Args:
            target_width (int): The baseline width images will be resized to before processing. Default is 800px.
        """
        self.target_width = target_width
        # Cache to store image properties for the fast validation step
        # We initialize these as empty/zero. Later on, we store image-wide calculations so we don't recompute them for every single coin candidate.
        self._val_gray = None
        self._val_h = self._val_w = 0
        self._val_bg_gray = 0.0
        self._val_is_coloured = False
        self._val_sat = None
        self._val_bg_sat = 0.0

    # PREPROCESSING
    def preprocess(self, img):
        """
        Resizes the image to a standard size to ensure consistent parameter tuning.
        
        Args:
            img (numpy.ndarray): The original image.
            
        Returns:
            tuple: A tuple containing:
                - resized_img (numpy.ndarray): the scaled image.
                - scale (float): the scaling factor applied.
        """
        h, w = img.shape[:2]
        scale = self.target_width / w
        new_w = self.target_width
        new_h = int(h * scale)
        
        # Hard cap on height: prevents massive performance drops if someone inputs a very tall, panoramic image.
        if new_h > 1000:
            scale = 1000.0 / h
            new_h = 1000
            new_w = int(w * scale)
            
        # INTER_AREA is a method for shrinking images in opencv as it prevents patterns and aliasing.
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale

    def analyse_background(self, img):
        """
        Samples the outer borders of the image to figure out background characteristics.
        
        Args:
            img (numpy.ndarray): The resized image.
            
        Returns:
            dict: A dictionary of background properties containing:
                - 'is_light' (bool): True if the background is mostly bright.
                - 'is_coloured' (bool): True if the background is highly saturated.
                - 'hue' (float): Median hue of the background.
                - 'sat' (float): Median saturation.
                - 'val' (float): Median brightness/value.
        """
        h, w = img.shape[:2]

        # Dynamic border thickness: 4% of the shortest side, bounded to a minimum of 5 pixels.
        b = max(int(min(h, w) * 0.04), 5)

        # Convert to HSV. It separates color intensity (Saturation) and brightness (Value)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Grab the pixel indices for the four edges (top, bottom, left, right).
        idx = np.concatenate([
            np.ravel_multi_index(np.mgrid[:b, :w], (h, w)).ravel(),
            np.ravel_multi_index(np.mgrid[h-b:h, :w], (h, w)).ravel(),
            np.ravel_multi_index(np.mgrid[:h, :b], (h, w)).ravel(),
            np.ravel_multi_index(np.mgrid[:h, w-b:w], (h, w)).ravel(),
        ])
        
        # Flatten the image to apply the 1D indices, extracting just the border pixels.
        flat = hsv.reshape(-1, 3)
        border = flat[idx]

        # We use median instead of mean. If a coin touches the edge, a mean would be skewed
        # Use median to avoid influence of outliers (like a coin touching the border)
        med_h, med_s, med_v = np.median(border, axis=0)
        
        return {
            'is_light': med_v > 127, # 127 is the middle of the 0-255 range
            'is_coloured': med_s > 115, # Tested threshold separating "colorful" from "drab/gray"
            'hue': med_h, 'sat': med_s, 'val': med_v,
        }

    # SEGMENTATION
    def _segment(self, img, bg_info):
        """
        Directs the image to the appropriate segmentation method based on the 
        background analysis performed earlier.
        
        Args:
            img (numpy.ndarray): The resized image.
            bg_info (dict): The dictionary containing background characteristics (generated by analyse_background method).
                            
        Returns:
            numpy.ndarray: A binary mask (2D array) where coins are white (255) and the background is black (0).
        """
        if bg_info['is_coloured']:
            return self._segment_coloured(img, bg_info)
        return self._segment_neutral(img, bg_info['is_light'])

    def _segment_neutral(self, img, is_light):
        """
        Binary segmentation for white, gray, or black backgrounds using grayscale intensity.Handles uneven lighting and shadows using adaptive thresholding.
        
        Args:
            img (numpy.ndarray): The resized image.
            is_light (bool): Flag indicating if the background is lighter than the coins.
            
        Returns:
            numpy.ndarray: A cleaned-up binary mask of the coins.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # It boosts local contrast without blowing out the entire image, helping to pull dark coins out of shadows.
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Strong blur to smooth out internal coin engravings (stars, faces, numbers).
        # When we reduce the blur, the thresholding will create Swiss cheese masks.
        blurred = cv2.GaussianBlur(enhanced, (13, 13), 3)
        
        # Adaptive Threshold
        # A coin in a dark corner and a coin under a bright lamp both get detected accurately.
        # We tested the simple global Otsu here, but it's not a good idea.
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV if is_light else cv2.THRESH_BINARY,
                                     21, 4) # BlockSize 21 (size of the local nighborhood must be odd)
        
        # Morphological operations to clean up the messy binary mask.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # CLOSE (Dilate then Erode): Fills in black holes inside the white coin blobs caused by harsh reflections.
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=4)

        # OPEN (Erode then Dilate): Wipes out tiny white noise dots (dust, scratches) in the black background without shrinking the actual coins.
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
        
        return binary

    def _segment_coloured(self, img, bg_info):
        """
        Segmentation designed for colored surfaces (e.g., a blue desk, a red book) 
        using the Saturation channel.
        
        Args:
            img (numpy.ndarray): The resized image.
            bg_info (dict): Background characteristics.
            
        Returns:
            numpy.ndarray: A cleaned-up binary mask of the coins.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Extract just the Saturation channel. Metal coins generally have very low saturation (they are grayish/silvery/brassy), while colored backgrounds have high saturation.
        # We don't use the Value channel for segmentation here because a white coin on a white background would be indistinguishable in terms of brightness, but it would still have low saturation compared to the colorful background.
        sat = hsv[:, :, 1]
        blurred_s = cv2.GaussianBlur(sat, (13, 13), 3)
        
        # Here we can use the Otsu's method, it automatically finds the best threshold value to split the bimodal histogram (low-saturation coins vs high-saturation background).
        # We invert it (THRESH_BINARY_INV) so the low-saturation coins become white (255).
        _, binary = cv2.threshold(blurred_s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up using the exact same morphological logic as the _segment_neutral method.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=4)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
        return binary

    # VALIDATION
    def _precompute_validation(self, img, bg_info):
        """
        Calculates and caches heavy image transformations (like grayscale and edge maps) once per image.
        This prevents recalculating them for every single coin candidate, massively speeding up the validation process.
        (We have ~3 min of execution without this method, and ~1s with it.)

        Args:
            img (numpy.ndarray): The resized image.
            bg_info (dict): Background characteristics from analyse_background().
            
        Returns:
            None: This method updates instance variables (self._val_*) defined in the constructor instead of returning.
        """
        h, w = img.shape[:2]

        # Cache dimensions and the grayscale version of the image
        self._val_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._val_h, self._val_w = h, w
        
        # Recreate the border mask to find the average background intensity
        bdr = max(int(min(h, w) * 0.04), 5)
        bm = np.zeros((h, w), dtype=np.uint8)
        bm[:bdr, :] = 255; bm[-bdr:, :] = 255; bm[:, :bdr] = 255; bm[:, -bdr:] = 255
        
        # Store the baseline background gray level
        self._val_bg_gray = cv2.mean(self._val_gray, mask=bm)[0]
        self._val_is_coloured = bg_info.get('is_coloured', False)
        
        # If the background is colored, we also cache its average saturation
        if self._val_is_coloured:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            self._val_sat = hsv[:, :, 1]
            # Calculate mean saturation of the borders
            self._val_bg_sat = cv2.mean(self._val_sat, mask=bm)[0]
            
        # NEW (19/02): This allows validating a coin even if it has the same color as the background
        # Thresholds 50/150 are standard for detecting metallic edges (it improved the results)
        self._val_edges = cv2.Canny(self._val_gray, 50, 150)

    def _validate_fast(self, x, y, r):
        """
        Checks if a mathematical circle actually contains a real coin by analyzing the image pixels locally (contrast and edges).
        
        Args:
            x (int): X-coordinate of the circle's center.
            y (int): Y-coordinate of the circle's center.
            r (int): Radius of the circle.
            
        Returns:
            bool: True if the candidate passes the tests (is likely a coin), False otherwise.
        """
        h, w = self._val_h, self._val_w
        if x - r < 4 or y - r < 4 or x + r > w - 4 or y + r > h - 4:
            return False

        # COLOR CRITERIA (The "Donut Test")
        # We compare the color inside the coin vs the color immediately outside it.

        # Create a filled circle mask slightly smaller than the coin itself (70% radius).
        # This guarantees we are only looking at the metal, avoiding the edge shadow.
        mask_in = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_in, (x, y), int(r * 0.7), 255, -1)
        
        # Create a "donut" mask representing the immediate background around the coin.
        mask_out = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_out, (x, y), int(r * 1.2), 255, -1) # (120% radius)
        cv2.circle(mask_out, (x, y), r, 0, -1) # Remove inside to get the ring

        # Calculate the average grayscale intensity for both regions
        mean_in = cv2.mean(self._val_gray, mask=mask_in)[0]
        mean_out = cv2.mean(self._val_gray, mask=mask_out)[0]
        diff_color = abs(mean_in - mean_out)

        # EDGE CRITERIA (New, added with the idea that even if a coin is perfectly camouflaged, it should still have a sharp edge detectable by Canny)

        # Create a mask that is just the thin outer perimeter of the circle (thickness 2)
        mask_edge = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_edge, (x, y), r, 255, 2) # Just the contour (thickness 2)
        
        # If the circle aligns with real edges in the image, this score goes up.
        edge_score = cv2.mean(self._val_edges, mask=mask_edge)[0]

        # DECISION
        
        # Case A : Strong color contrast (Gold coin on Gray background)
        # A shiny gold coin on a dark wooden table. 
        # The donut test alone is enough to prove it's an object.
        if diff_color > 15: 
            return True
            
        # Case B : The Camouflage scenario. (Gray coin on Gray background)
        # A silver coin on a gray table. `diff_color` might be very low.
        # Threshold 30 corresponds to approx 12% of perimeter detected as "strong edge".
        # But the physical thickness of the coin casts a tiny shadow, creating a sharp edge.
        # A score > 30 means roughly 12% of the circle's perimeter matched a sharp Canny edge.
        if edge_score > 30: 
            return True

        # If it fails both tests, it's a false positive (like flowers images of the group 5)
        return False

    # HOUGH TRANSFORM
    def _detect_hough(self, img, gray_blur=None):
        """
        Finds geometric circles in the image using the Hough Gradient method.
        Instead of relying on a single hardcoded threshold, it sweeps through 
        multiple sensitivity levels to find the most reliable detections.
        
        Args:
            img (numpy.ndarray): The resized BGR image.
            gray_blur (numpy.ndarray, optional): A pre-computed blurred grayscale image. If None, it will be generated.
                                                 
        Returns:
            list: A list of dictionaries, where each dict represents a validated coin: [{'x': center_x, 'y': center_y, 'r': radius}, ..]
        """

        # PREPARATION
        if gray_blur is None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Boost contrast to make the outer edges of the coins "pop"
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # New : (15, 15) massive blur specifically for Hough to ignore internal details
            gray_blur = cv2.GaussianBlur(clahe.apply(gray), (15, 15), 3)
            
        h, w = img.shape[:2]
        min_dim = min(h, w)

        # DYNAMIC SIZING
        # Calculate expected minimum and maximum radii based on image size.
        min_r, max_r = max(int(min_dim * 0.04), 22), int(min_dim * 0.28)
        
        # Set the minimum distance between circle centers. 
        # Multiplier 2.1 ensures that two detected centers can't belong to the same coin (even if we have this example in the dataset). The absolute minimum is 55 pixels to prevent too many false circles in very small images.
        min_dist = max(int(min_r * 2.1), 55)

        # PARAMETER SWEEP & DETECTION
        # p1: The upper threshold for the internal Canny edge detector.
        # p2: The accumulator threshold (how "perfect" the circle needs to be).
        # We start strict (120, 55). If nothing passes, we relax the rules.
        for p1, p2 in [(120, 55), (100, 45), (85, 35)]:

            circles = cv2.HoughCircles(
                gray_blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2, # Inverse ratio of the accumulator resolution to image resolution
                minDist=min_dist,
                param1=p1,
                param2=p2,
                minRadius=min_r,
                maxRadius=max_r
            )
            
            if circles is not None:
                out = []
                for c in np.round(circles[0]).astype(int):
                    if self._validate_fast(c[0], c[1], c[2]):
                        out.append({'x': int(c[0]), 'y': int(c[1]), 'r': int(c[2])})
                
                # If we found and validated at least one coin at this strictness level, clean up any overlapping bounding boxes with NMS and return immediately.
                if out: 
                    return self._nms(out, ratio=0.85)
        return [] # If all parameter levels fail, return an empty list

    # WATERSHED
    def _detect_watershed(self, img, binary):
        """
        Separates touching or slightly overlapping coins by treating the binary mask as a topological map and 'flooding' it using the Watershed algorithm.
        
        Args:
            img (numpy.ndarray): The resized original image (now must be 3 channels for Watershed).
            binary (numpy.ndarray): The cleaned binary mask from the segmentation phase.
            
        Returns:
            list: A list of dictionaries representing validated coins: [{'x': center_x, 'y': center_y, 'r': radius}, ...]
        """
        # if the mask is completely black, there's nothing to process.
        if binary is None or np.sum(binary) == 0: return []
        
        # Calculates the distance from every white pixel to the nearest black pixel.
        # The center of a coin is furthest from the edge, so it becomes the brightest spot (the peak).
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # FINDING 'SURE' FOREGROUND
        # We aggressively threshold the distance map to isolate just the centers of the coins.
        # We think these values are the "sure foreground" because they are very unlikely to be anything other than the middle of a coin.
        thresh = max(0.28, 0.30 * dist_norm.max()) 
        _, sure_fg = cv2.threshold(dist_norm, thresh, 1.0, 0)
        sure_fg = np.uint8(sure_fg * 255)
        
        # FINDING THE UNKNOWN BORDER REGIONS
        # Subtract the sure foreground (the centers) from the dilated mask.
        unknown = cv2.subtract(cv2.dilate(binary, None, iterations=3), sure_fg)
        
        # Give every isolated coin center a unique label (1, 2, 3, etc.)
        num_labels, markers = cv2.connectedComponents(sure_fg)

        # Watershed requires the background to be exactly 1, and the unknown regions to be 0.
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # THE WATERSHED ALGORITHM
        img_3ch = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # When water from two different coins meets, it builds a 'dam' (marked as -1), cleanly separating them.
        markers = cv2.watershed(img_3ch, markers)
        
        # EXTRACTION AND VALIDATION
        coins = []
        for label in range(2, num_labels + 1):

            mask = np.uint8(markers == label) * 255
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not cnts: 
                continue

            # Grab the largest contour in case of minor fragmentation
            cnt = max(cnts, key=cv2.contourArea)

            # Noise filter: if the segmented area is smaller than a tiny circle (radius 18), ignore it.
            if cv2.contourArea(cnt) < (np.pi * 18**2): 
                continue
            
            # Convert the irregular blob shape back into a perfect geometric circle
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)

            # Run it through the physical validation tests (Donut Test / Edge Check)
            if self._validate_fast(int(cx), int(cy), int(radius)):
                coins.append({'x': int(cx), 'y': int(cy), 'r': int(radius)})
        return coins

    # CONTOUR DETECTION
    def _detect_contour_simple(self, img, binary):
        """
        Detects coins by finding external boundaries (contours) in the binary mask and evaluating their geometric properties (circularity and solidity).
        
        Args:
            img (numpy.ndarray): The resized image.
            binary (numpy.ndarray): The cleaned binary mask from the segmentation phase.
            
        Returns:
            list: The list of dictionaries representing validated coins: [{'x': center_x, 'y': center_y, 'r': radius}, ...]
        """
        # RETR_EXTERNAL ignores holes inside the coins and only grabs the outer boundary.
        # CHAIN_APPROX_SIMPLE compresses horizontal/vertical lines to save memory.
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coins = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perim = cv2.arcLength(cnt, True)

            if perim == 0: # Division by zero
                continue
            
            # Circularity metric: 4*pi*Area / Perimeter^2 (1.0 = perfect circle)
            circ = 4 * np.pi * area / (perim ** 2)
            if circ < 0.72: continue # Strict circularity check
            
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            # Solidity check: ensures the coin isn't just a crescent or ring shape
            if area / (np.pi * r**2) < 0.75: continue
            
            # Pass the candidate to our physical pixel validator
            if self._validate_fast(int(cx), int(cy), int(r)):
                coins.append({'x': int(cx), 'y': int(cy), 'r': int(r)})
        return coins

    #################
    # POST-PROCESSING
    ###############
    @staticmethod
    def _nms(coins, ratio=0.85):
        """
        Non-Maximum Suppression (NMS): Removes overlapping circle detections.
        If two algorithms detect the same coin slightly differently, this keeps 
        the largest detection and deletes the duplicate.
        """
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
        """
        Removes outlier detections based on real-world Euro coin proportions.Relies on the constant MIN_RADIUS_RATIO (approx 0.52).
        """
        # We need at least 3 coins to establish a reliable median size.
        if len(coins) <= 2: return coins
        radii = [c['r'] for c in coins]

        med = float(np.median(radii))

        lo, hi = med * MIN_RADIUS_RATIO, med / MIN_RADIUS_RATIO

        return [c for c in coins if lo <= c['r'] <= hi]

    ##################
    # DETECTION PIPELINE
    ##################
    def detect(self, img, bg_info):
        """
        The main orchestrator. Runs all segmentation and detection strategies, merges their results, and applies final filtering.
        
        Args:
            img (numpy.ndarray): The resized original image.
            bg_info (dict): Background characteristics from analyse_background().
            
        Returns:
            list: The final, validated, and filtered list of detected coins.
        """
        # 1. Setup and Preprocessing for Validation (heavy transformations are already cached for speed)
        self._precompute_validation(img, bg_info)
        binary = self._segment(img, bg_info)
        
        # Prepare the specific blurred image required by the Hough Transform
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_blur = cv2.GaussianBlur(clahe.apply(self._val_gray), (15, 15), 3)

        # 2. Run all three detection strategies independently
        coins_ws = self._detect_watershed(img, binary)
        coins_cs = self._detect_contour_simple(img, binary)
        coins_h  = self._detect_hough(img, gray_blur)

        # 3. Merging Strategy
        # We start with Hough. Because it relies on strict geometry, if Hough finds a coin, 
        # it's usually highly accurate in terms of exact center and radius.
        merged = list(coins_h)
        for other in [coins_ws, coins_cs]:
            for oc in other:
                # Add detection only if it doesn't significantly overlap a piece already found
                # Inclusion check: prevents counting internal stars/features
                if not any(np.hypot(oc['x']-m['x'], oc['y']-m['y']) < max(oc['r'], m['r']) * 0.85 for m in merged):
                    merged.append(oc)

        # 4. Final Cleanup
        # Run NMS one last time just in case the merge introduced slight overlaps
        merged = self._nms(merged, ratio=0.85)

        # 5. Filter out logically impossible sizes and return the final list
        return self._filter_radii(merged)