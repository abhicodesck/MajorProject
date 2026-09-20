import cv2
import numpy as np


def gamma_correction(image,gamma=1.5):


    #Create look up table
    table = np.array([
        ((i/255.0)**(1/gamma))*255
        for i in np.arange(0,256)
    ]).astype("uint8")

    #applying gamma correction

    corrected = cv2.LUT(image,table)

    return corrected

def apply_clahe(image):

    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Apply CLAHE to L channel
    l = clahe.apply(l)

    # Merge channels
    enhanced_lab = cv2.merge((l, a, b))

    # Convert LAB back to BGR
    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced