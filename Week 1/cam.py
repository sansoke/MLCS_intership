# Read the given image (person.jpg) and make the photo look straight.
# Use rotate method.
# Resize the given image into 50%.
# Use openCV and dlib libraries to extract the facial landmarks from the given image.
# Link to download the landmark data: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# Crop the area in the image that contains the face.
# Make sure that the cropped area contains the all of the facial landmarks.
# Read the sunglasses image (sunglasses.png) and resize it to fit the face image that you cropped.
# Note that the sunglasses image is in png format (4 channels).
# Put the sunglasses on a face (cropped image).
# Make sure that the transparent background of the sunglasses image is preserved.
# Hint : replace the pixel information at the desired position with the sunglasses image.

import cv2
import numpy as np
import dlib
from djitellopy import Tello

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib

# Load sunglasses image with alpha channel
sunglasses = cv2.imread("sunglasses.png", cv2.IMREAD_UNCHANGED)

import cv2
from djitellopy import Tello

# Initialize Tello
tello = Tello()
tello.connect()
tello.streamon()

# Get the video stream
frame_reader = tello.get_frame_read()

frame = frame_reader.frame
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imshow("frame", rgb)


while True:    
    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get eye coordinates
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        # Compute width & height for sunglasses
        glasses_width = int((right_eye[0] - left_eye[0]))
        glasses_height = int(glasses_width * (sunglasses.shape[0] / sunglasses.shape[1])) # Maintain aspect ratio

        # Resize sunglasses
        sunglasses_resized = cv2.resize(sunglasses, (glasses_width, glasses_height))

        # Positioning the sunglasses
        x_offset = left_eye[0]
        y_offset = left_eye[1] - (glasses_height // 2)  # Adjust to fit naturally
        
        # Overlay sunglasses using alpha blending
        for i in range(glasses_height):
            for j in range(glasses_width):
                if y_offset + i >= frame.shape[0] or x_offset + j >= frame.shape[1]:
                    continue
                alpha = sunglasses_resized[i, j, 3] / 255.0  # Normalize alpha
                for c in range(3):  # Blend BGR channels
                    frame[y_offset + i, x_offset + j, c] = (
                        (1 - alpha) * frame[y_offset + i, x_offset + j, c] +
                        alpha * sunglasses_resized[i, j, c]
                    )

    # Show live feed with overlay
    cv2.imshow("Live Sunglasses Overlay", frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imwrite("sunglasses_overlay.png", frame)
# Cleanup

tello.streamoff()
tello.end()
cv2.destroyAllWindows()
