import face_recognition
import cv2
import numpy as np


def get_hair_color(image, face_location):
    # Define the region above the face to estimate hair color
    top, right, bottom, left = face_location
    head_region = image[top - 100:top, left:right]  # Look above the face for hair

    # Convert to HSV to make color detection easier
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)

    # Threshold for possible hair colors in the HSV color space
    # Example: We will look for colors that resemble black or brown hair
    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([15, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    # Mask the brown and black areas
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Count the number of brown and black pixels
    brown_pixels = cv2.countNonZero(brown_mask)
    black_pixels = cv2.countNonZero(black_mask)

    # Compare counts to guess the hair color
    if brown_pixels > black_pixels:
        return "Brown Hair"
    elif black_pixels > brown_pixels:
        return "Black Hair"
    else:
        return "Uncertain Hair Color"


# Load an image file
image_path = 'C:\\Users\\ailiv\\Desktop\\mix.jpg'
image = cv2.imread(image_path)

# Find all faces in the image
face_locations = face_recognition.face_locations(image, model="cnn", number_of_times_to_upsample=1)
#face_locations = face_recognition.face_locations(image, model="cnn")
for face_location in face_locations:
    # Get the face location
    print(f"Face found at {face_location}")

    # Get the estimated hair color
    hair_color = get_hair_color(image, face_location)
    print(f"Estimated hair color: {hair_color}")

    # Draw a box around the face
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 5)

# Display the image with the face and hair color estimation
cv2.imshow("Face and Hair Color", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
