import cv2
import numpy as np


# --------------------------------
# 1. Gamma Correction
# --------------------------------

def gamma_correction(image, gamma=1.2):
    """
    Brightens a dark image while preserving its dimensions.

    gamma > 1 -> brighter
    gamma < 1 -> darker
    """

    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


# --------------------------------
# 2. CLAHE Enhancement
# --------------------------------

def apply_clahe(image):
    """
    Improves local contrast using CLAHE.
    CLAHE is applied only to the luminance channel.
    """

    # BGR -> LAB
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    # Split channels
    l, a, b = cv2.split(lab)

    # Create CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Apply CLAHE to brightness channel
    l = clahe.apply(l)

    # Merge channels
    enhanced_lab = cv2.merge(
        (l, a, b)
    )

    # LAB -> BGR
    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# --------------------------------
# 3. Mild Denoising
# --------------------------------

def denoise(image):
    """
    Reduces noise while preserving object details.
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
# 4. Complete Enhancement Pipeline
# --------------------------------

def enhance_night_image(image):
    """
    Complete night-time enhancement pipeline.

    Steps:
    1. Gamma correction
    2. CLAHE
    3. Mild denoising
    """

    # Step 1: Brightness enhancement
    gamma_image = gamma_correction(
        image,
        gamma=1.2
    )

    # Step 2: Contrast enhancement
    clahe_image = apply_clahe(
        gamma_image
    )

    # Step 3: Noise reduction
    final_image = denoise(
        clahe_image
    )

    return final_image


# --------------------------------
# 5. Test the Enhancement
# --------------------------------

if __name__ == "__main__":

    # Change this to your image path
    input_path = "C:/SmartVisionSystem/data/input/image1.jpg"

    # Output path
    output_path = (
        "C:/SmartVisionSystem/data/output/enhanced_night.jpg"
    )

    # Read image
    image = cv2.imread(input_path)

    if image is None:
        print("Error: Could not load image.")
        exit()

    # Enhance image
    enhanced_image = enhance_night_image(image)

    # --------------------------------
    # Save enhanced image
    # --------------------------------

    success = cv2.imwrite(
        output_path,
        enhanced_image
    )

    if success:
        print("Enhanced image saved successfully:")
        print(output_path)
    else:
        print("Error: Could not save enhanced image.")

    # --------------------------------
    # Display Original
    # --------------------------------

    cv2.namedWindow(
        "Original Image",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "Original Image",
        1000,
        600
    )

    cv2.imshow(
        "Original Image",
        image
    )

    # --------------------------------
    # Display Enhanced
    # --------------------------------

    cv2.namedWindow(
        "Enhanced Image",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "Enhanced Image",
        1000,
        600
    )

    cv2.imshow(
        "Enhanced Image",
        enhanced_image
    )

    # --------------------------------
    # Wait
    # --------------------------------

    print("\nPress any key to close the images.")

    cv2.waitKey(0)

    cv2.destroyAllWindows()