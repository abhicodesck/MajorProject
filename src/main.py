import cv2

from enhancement import enhance_night_image


# Read image
image = cv2.imread("C:/SmartVisionSystem/data/input/image1.jpg")

if image is None:
    print("Image not found!")
    exit()


# Apply gamma correction
gamma_image = enhance_night_image(image)


# Apply CLAHE
enhanced_image = apply_clahe(gamma_image)


# Save results
cv2.imwrite(
    "../data/output/gamma.jpg",
    gamma_image
)

cv2.imwrite(
    "../data/output/enhanced.jpg",
    enhanced_image
)


# Display
cv2.imshow("Original", image)
cv2.imshow("Gamma Corrected", gamma_image)
cv2.imshow("Enhanced", enhanced_image)

cv2.waitKey(0)
cv2.destroyAllWindows()