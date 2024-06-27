from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2 
import numpy as np
import mediapipe as mp 

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

def detect_ok_sign(hand_landmarks):
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            thumb_tip = hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            if (abs(thumb_tip.x - index_tip.x) < 0.02 and
                abs(thumb_tip.y - index_tip.y) < 0.02 and
                index_tip.y < index_mcp.y):
                return True
    return False

while True:
    ret, image = camera.read()
    if not ret:
        break

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Check if an OK sign is detected
    ok_sign_detected = detect_ok_sign(results.multi_hand_landmarks)

    if ok_sign_detected:
        cv2.putText(image, "OK Sign Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Not OK Sign", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()