import cv2
import numpy as np


# --------------------------------
# Gamma Correction
# --------------------------------

def gamma_correction(image, gamma=1.5):
    """
    Brightens a dark image while preserving natural colors.

    gamma > 1 -> brighter
    gamma < 1 -> darker
    """

    # Convert gamma value into exponent
    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


# --------------------------------
# CLAHE Enhancement
# --------------------------------

def apply_clahe(image):
    """
    Applies CLAHE only to the luminance channel.
    This prevents color distortion.
    """

    # Convert BGR -> LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab)

    # CLAHE configuration
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Apply CLAHE ONLY to brightness channel
    l = clahe.apply(l)

    # Merge channels
    enhanced_lab = cv2.merge((l, a, b))

    # Convert LAB -> BGR
    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# --------------------------------
# Mild Denoising
# --------------------------------

def denoise(image):
    """
    Removes some noise without destroying object details.
    """

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        3,
        3,
        7,
        21
    )


# --------------------------------
# Complete Night Enhancement
# --------------------------------

def enhance_night_image(image):
    """
    Complete night image enhancement pipeline.
    """

    # Step 1: Brighten
    gamma_image = gamma_correction(
        image,
        gamma=1.5
    )

    # Step 2: Improve local contrast
    clahe_image = apply_clahe(
        gamma_image
    )

    # Step 3: Mild noise reduction
    final_image = denoise(
        clahe_image
    )

    return final_image