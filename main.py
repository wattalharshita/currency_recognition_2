import cv2

# Load the image
image = cv2.imread('note.jpg')

# Check if image loaded properly
if image is None:
    print("Error: Image not found or couldn't be loaded.")
else:
    # Show the image in a window
    cv2.imshow('Currency Note', image)

    # Wait until any key is pressed
    cv2.waitKey(0)

    # Close the window
    cv2.destroyAllWindows()


import cv2

# Load the image
image = cv2.imread('note.jpg')

# Convert it to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the grayscale image
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 



# Apply binary thresholding
_, threshold_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Show the thresholded image
cv2.imshow('Thresholded Image', threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Find contours in the thresholded image
contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
contoured_image = image.copy()
cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 3)  # Green contours

# Show the image with contours
cv2.imshow('Contours', contoured_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
