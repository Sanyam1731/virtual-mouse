import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

screen_width, screen_height = pyautogui.size()

cv2.namedWindow("Hand Gesture Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Gesture Detection", cv2.WND_PROP_TOPMOST, 1)

# Add state for pinch gesture
pinch_active = False

while True:
    ret, image = camera.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Flip the image for a mirror view
    image = cv2.flip(image, 1)

    # Get and display image width and height
    height, width, _ = image.shape
    cv2.putText(
        image,
        f"Width: {width}, Height: {height}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image and find hands
    results = hands.process(rgb_image)

    # Debug: Print if hands are detected
    if results.multi_hand_landmarks:
        print("Hand detected!")
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip and thumb tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Convert normalized coordinates to image coordinates
            ix, iy = int(index_finger_tip.x * width), int(index_finger_tip.y * height)
            tx, ty = int(thumb_tip.x * width), int(thumb_tip.y * height)

            # Draw circles on tips
            cv2.circle(image, (ix, iy), 10, (0, 255, 255), -1)
            cv2.circle(image, (tx, ty), 10, (0, 0, 255), -1)

            # Stabilize mouse movement by smoothing
            if 'prev_screen_x' not in globals():
                global prev_screen_x, prev_screen_y
                prev_screen_x, prev_screen_y = 0, 0
            screen_x = np.interp(index_finger_tip.x, [0, 1], [0, screen_width])
            screen_y = np.interp(index_finger_tip.y, [0, 1], [0, screen_height])
            # Simple smoothing
            smooth_x = prev_screen_x + (screen_x - prev_screen_x) * 0.7
            smooth_y = prev_screen_y + (screen_y - prev_screen_y) * 0.7
            pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)
            prev_screen_x, prev_screen_y = smooth_x, smooth_y

            # Mouse click/drag if index and thumb tips are close
            distance = np.hypot(ix - tx, iy - ty)
            pinch_threshold = 40  # You can adjust this threshold
            if distance < pinch_threshold:
                if not pinch_active:
                    # Pinch started: mouse down (for drag/click)
                    pyautogui.mouseDown()
                    pinch_active = True
                    cv2.putText(image, "Click/Drag!", (ix, iy-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                if pinch_active:
                    # Pinch released: mouse up (release drag)
                    pyautogui.mouseUp()
                    pinch_active = False

    cv2.imshow("Hand Gesture Detection", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
