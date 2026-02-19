"""
Coin Identification Module
Classifies detected coins by color group (copper / gold / bimetallic) and assigns exact Euro values based purely on image processing rules (with an optional ML test mode).

How it works:
- Reference Data: Stores the official diameters and colors of all Euro coins.
- Feature Extraction: Analyzes the pixels inside each coin to get average colors (using HSV and LAB spaces).
- Bimetallic Score: Looks for a sharp color contrast between the center and the outer ring to spot 1€ and 2€ coins.
- Rule-Based Logic: Uses specific color math (depending on whether the background is dark, light, or colored) to guess the metal.
- Relative Sizing: Finds the biggest coin in the image, assumes its real-world size, and uses it as a ruler to calculate the value of all other coins.
"""

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Official specifications of Euro coins. We use this as our absolute reference.
COIN_SPECS = {
    '1c':    {'diameter_mm': 16.25, 'color_group': 'copper',     'value': 0.01},
    '2c':    {'diameter_mm': 18.75, 'color_group': 'copper',     'value': 0.02},
    '5c':    {'diameter_mm': 21.25, 'color_group': 'copper',     'value': 0.05},
    '10c':   {'diameter_mm': 19.75, 'color_group': 'gold',       'value': 0.10},
    '20c':   {'diameter_mm': 22.25, 'color_group': 'gold',       'value': 0.20},
    '50c':   {'diameter_mm': 24.25, 'color_group': 'gold',       'value': 0.50},
    '1euro': {'diameter_mm': 23.25, 'color_group': 'bimetallic', 'value': 1.00},
    '2euro': {'diameter_mm': 25.75, 'color_group': 'bimetallic', 'value': 2.00},
}

# Helper lists to quickly determine which values belong to which color family
COPPER_DENOMS  = ['1c', '2c', '5c']
GOLD_DENOMS    = ['10c', '20c', '50c']
BIMETAL_DENOMS = ['1euro', '2euro']

# In the photo, we rely on the fact that the biggest copper coin must be a 5c coin, etc.
LARGEST_BY_COLOR = {'copper': '5c', 'gold': '50c', 'bimetallic': '2euro'}


class CoinIdentification:
    """Classifies coins by color and assigns Euro denominations based on relative size."""

    def __init__(self, use_knn=True):
        self.use_knn = use_knn
        self.knn = self.scaler = None

        # Initialize the KNN classifier if the option is enabled
        # (NOT USED IN THE FINAL VERSION, BUT CAN BE ACTIVATED FOR TESTING PURPOSES).
        if use_knn:
            self._build_knn()

    # KNN CLASSIFIER
    def _build_knn(self):
        """
        Trains a small K-Nearest Neighbors model on representative coin color features.
        
        Args:
            None
            
        Returns:
            None (Modifies self.knn and self.scaler in-place).
        """
        # Hardcoded representative feature vectors for various lighting/backgrounds
        X = np.array([
            # COPPER - light bg
            [12,56,177,134,139,22],[13,113,141,138,148,25],[14,90,155,136,144,23],
            [11,70,165,135,141,21],[15,105,148,137,146,24],
            # COPPER - dark bg
            [16,210,158,147,174,17],[17,209,178,147,180,8],[17,211,167,145,178,13],
            [15,200,150,146,172,15],[18,215,170,144,176,12],
            # GOLD - light bg
            [24,93,139,126,150,25],[24,108,108,126,149,26],[25,100,125,127,148,24],
            [23,95,130,125,147,25],[26,110,120,128,151,23],
            # GOLD - dark bg
            [21,161,180,133,173,15],[21,203,200,136,189,16],[20,140,175,132,166,17],
            [21,182,191,133,181,18],[21,208,186,136,185,21],
            # BIMETALLIC - light bg
            [52,88,134,131,128,35],[28,64,101,126,140,29],[24,35,123,127,135,23],
            [21,62,101,128,137,32],[25,50,115,128,136,28],
            # BIMETALLIC - dark bg
            [21,155,165,132,170,18],[20,160,175,133,172,16],[21,150,160,131,168,19],
            [20,162,187,134,175,18],[21,140,170,130,165,17],
        ], dtype=float)
        
        # Corresponding labels for the training set
        y = np.array(['copper']*10 + ['gold']*10 + ['bimetallic']*10)
        
        # Setup data scaler to normalize feature ranges
        self.scaler = StandardScaler()
        # Initialize KNN with 5 neighbors and distance-based weights
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        # Train the model
        self.knn.fit(self.scaler.fit_transform(X), y)

    #  COLOUR FEATURE EXTRACTION
    @staticmethod
    def _color_features(hsv, lab, x, y, r):
        """
        Extracts HSV/Lab mean colors and saturation standard deviation from a coin region.
        
        Args:
            hsv (numpy.ndarray): The full image in HSV color space.
            lab (numpy.ndarray): The full image in LAB color space.
            x (int): X-coordinate of the coin's center.
            y (int): Y-coordinate of the coin's center.
            r (int): Radius of the coin in pixels.
            
        Returns:
            numpy.ndarray: An array containing [H_mean, S_mean, V_mean, a*_mean, b*_mean, S_std].
        """
        h, w = hsv.shape[:2]

        # Create a mask for the circular coin area (~78% of radius to avoid edge noise)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), max(int(r * 0.78), 5), 255, -1)
        
        # Calculate mean values for HSV and Lab color spaces
        h_m, s_m, v_m, _ = cv2.mean(hsv, mask=mask)
        _, a_m, b_m, _ = cv2.mean(lab, mask=mask)
        
        # Extract individual saturation pixels to compute standard deviation
        s_pix = hsv[:, :, 1][mask > 0]
        s_std = float(np.std(s_pix)) if len(s_pix) > 0 else 0.0
        
        # Return the feature vector
        return np.array([h_m, s_m, v_m, a_m, b_m, s_std])

    @staticmethod
    def _enh_bimetallic(lab, x, y, r):
        """
        Calculates an enhanced bimetallic score by finding the maximum LAB color 
        difference between the inner core and outer ring of the coin.
        
        Args:
            lab (numpy.ndarray): The full image in LAB space.
            x (int): X-coordinate of the coin's center.
            y (int): Y-coordinate of the coin's center.
            r (int): Radius of the coin in pixels.
            
        Returns:
            float: The maximum color contrast found (higher means likely bimetallic).
        """
        h, w = lab.shape[:2]
        mx = 0.0
        # Sweep through several potential inner/outer boundaries to find the highest contrast
        for ratio in [0.42, 0.50, 0.58, 0.66, 0.74]:
            # Define inner core radius and outer ring bounds
            ir = max(int(r*(ratio-0.07)), 2)
            oi, oo = int(r*(ratio+0.05)), int(r*min(ratio+0.22, 0.88))
            
            # Create masks for the center core
            mi = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mi, (x, y), ir, 255, -1)
            
            # Create mask for the outer ring
            mo = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mo, (x, y), oo, 255, -1)
            cv2.circle(mo, (x, y), oi, 0, -1)
            
            # Compute Euclidean distance between average Lab colors of core and ring
            d = float(np.linalg.norm(
                np.array(cv2.mean(lab, mask=mi)[:3]) -
                np.array(cv2.mean(lab, mask=mo)[:3])))
            
            # Track the maximum color difference found
            mx = max(mx, d)
        return mx

    # RULE-BASED CLASSIFICATION (per background type)
    @staticmethod
    def _classify_dark(f, eb):
        """
        Classifies the coin color group on a dark, neutral background.
        
        Args:
            f (numpy.ndarray): The feature array [H, S, V, a, b, S_std].
            eb (float): The calculated bimetallic score.
            
        Returns:
            str: 'copper', 'gold', or 'bimetallic'.
        """
        # Use Lab a* channel to identify copper (reddish hue)
        if f[3] > 140: return 'copper'
        # Check bimetallic score for 1€/2€ coins
        if eb > 10:    return 'bimetallic'
        # Default to gold group
        return 'gold'

    @staticmethod
    def _classify_light(f, eb):
        """
        Classifies the coin color group on a light, neutral background.
        
        Args:
            f (numpy.ndarray): The feature array [H, S, V, a, b, S_std].
            eb (float): The calculated bimetallic score.
            
        Returns:
            str: 'copper', 'gold', or 'bimetallic'.
        """
        h_m, s_m, v_m, a_m, b_m, s_std = f
        # High bimetallic contrast or low saturation/value points to bimetallic
        if eb > 14:
            return 'bimetallic'
        if s_m < 55 and v_m < 130:
            return 'bimetallic'
        
        # High saturation variability (s_std) often appears in 1€/2€ patterns
        if s_std > 30 and s_m < 85 and eb>8:
            return 'bimetallic'
        
        # Color thresholds based on Lab a* and HSV Hue
        if a_m > 132 and h_m < 17:
            return 'copper'
        if a_m > 136:
            return 'copper'
        if b_m > 144 and s_m > 75:
            return 'gold'
        
        # Final fallback comparing color intensity distance from neutral gray (128)
        return 'copper' if (a_m-128) > (b_m-128)+3 else 'gold'

    def _classify_coloured_bg(self, f, eb):
        """
        Classifies the coin color group on a coloured background (e.g., a red table).
        
        Args:
            f (numpy.ndarray): The feature array [H, S, V, a, b, S_std].
            eb (float): The calculated bimetallic score.
            
        Returns:
            str: 'copper', 'gold', or 'bimetallic'.
        """
        h_m, s_m, v_m, a_m, b_m, s_std = f
        # Check for bimetallic transition first
        if eb > 12:   return 'bimetallic'
        # Strong a* indicates copper
        if a_m > 138:  return 'copper'
        # High b* combined with low a* points to gold
        if b_m > 150 and a_m < 135: return 'gold'
        # Fall back to dark background logic
        return self._classify_dark(f, eb)

    # SINGLE-COIN CLASSIFICATION
    def classify_one(self, hsv, lab, coin, bg_info):
        """
        Classifies a single coin by determining its color group.
        Combines rule-based classification with optional KNN refinement.
        
        Args:
            hsv (numpy.ndarray): The full image in HSV.
            lab (numpy.ndarray): The full image in LAB.
            coin (dict): The dictionary containing 'x', 'y', 'r'.
            bg_info (dict): The background analysis from the detection phase.
            
        Returns:
            dict: The modified coin dictionary containing its color properties.
        """
        x, y, r = coin['x'], coin['y'], coin['r']
        # Extract visual features and structural bimetallic score
        feats = self._color_features(hsv, lab, x, y, r)
        eb = self._enh_bimetallic(lab, x, y, r)

        # Select the rule-based decision tree based on environment analysis
        if bg_info['is_coloured']:
            cr = self._classify_coloured_bg(feats, eb)
        elif not bg_info['is_light']:
            cr = self._classify_dark(feats, eb)
        else:
            cr = self._classify_light(feats, eb)

        color = cr
        # Refine prediction with KNN if confidence is high enough
        if self.use_knn and self.knn is not None:
            X = self.scaler.transform(feats.reshape(1, -1))
            knn_pred = self.knn.predict(X)[0]
            conf = max(self.knn.predict_proba(X)[0])
            
            if conf > 0.65:
                # Prioritize strong structural bimetallic/copper signals over KNN
                if eb > 18:
                    color = 'bimetallic'
                elif feats[3] > 142:
                    color = 'copper'
                else:
                    color = knn_pred

        # Save classification metadata to the coin object
        coin.update(features=feats, enh_bimetallic_score=eb,
                    color_group=color, color_rules=cr)
        return coin

    # DENOMINATION ASSIGNMENT
    def assign_denominations(self, coins):
        """
        Assigns the exact Euro denomination to each coin based on its color group and relative size (pixel radius vs. known mm diameters).
        
        Args:
            coins (list): A list of coin dictionaries with 'color_group' assigned.
            
        Returns:
            list: The fully updated list of coins with 'denomination' and 'value'.
        """
        if not coins:
            return coins

        # Identify the largest coin to serve as a world-scale reference
        max_r = max(c['r'] for c in coins)
        largest = next(c for c in coins if c['r'] == max_r)
        # Determine its likely real-world diameter based on identified group
        ref = LARGEST_BY_COLOR.get(largest['color_group'], '50c')
        # Calculate pixel-to-millimeter ratio
        px_mm = (max_r * 2) / COIN_SPECS[ref]['diameter_mm']

        for c in coins:
            
            # Calculate estimated real-world diameter
            dia = (c['r'] * 2) / px_mm
            c['estimated_dia_mm'] = round(dia, 2)
            
            # Select the list of candidate denominations based on the coin's group
            cands = (COPPER_DENOMS if c['color_group'] == 'copper'
                     else GOLD_DENOMS if c['color_group'] == 'gold'
                     else BIMETAL_DENOMS)
            
            # Match to the official diameter that is mathematically closest
            best = min(cands, key=lambda d: abs(COIN_SPECS[d]['diameter_mm']-dia))

            c['denomination'] = best
            c['value'] = COIN_SPECS[best]['value']
        return coins