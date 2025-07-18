import string
import cv2
import pyautogui
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# Get frame dimensions
width = int(cap.get(3))
height = int(cap.get(4))

# Define QWERTY keyboard layout
keyboard_rows = [
    list('1234567890'),
    list('qwertyuiop'),
    list('asdfghjkl'),
    list('zxcvbnm'),
    ['SPACE', 'BACKSPACE', 'ENTER']
]
key_width = 80
key_height = 80
keyboard_y = height - (len(keyboard_rows) * (key_height + 10)) - 20

# State for pinch gesture
pinch_active = False
last_typed = ''

while True:
    success, image = cap.read()
    if not success:
        break
        
    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect hands
    results = hands.process(rgb_image)
    
    # Initialize finger positions
    ix, iy = 0, 0  # Index finger
    tx, ty = 0, 0  # Thumb

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip (landmark 8) and thumb tip (landmark 4)
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            # Convert to pixel coordinates
            ix, iy = int(index_tip.x * width), int(index_tip.y * height)
            tx, ty = int(thumb_tip.x * width), int(thumb_tip.y * height)
            
            # Draw circles on finger tips
            cv2.circle(image, (ix, iy), 10, (255, 0, 0), -1)
            cv2.circle(image, (tx, ty), 10, (0, 255, 0), -1)

            # Move the mouse cursor
            screen_width, screen_height = pyautogui.size()
            screen_x = np.interp(index_tip.x, [0, 1], [0, screen_width])
            screen_y = np.interp(index_tip.y, [0, 1], [0, screen_height])
            if 'prev_screen_x' not in globals():
                global prev_screen_x, prev_screen_y
                prev_screen_x, prev_screen_y = 0, 0
            smooth_x = prev_screen_x + (screen_x - prev_screen_x) * 0.7
            smooth_y = prev_screen_y + (screen_y - prev_screen_y) * 0.7
            pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)
            prev_screen_x, prev_screen_y = smooth_x, smooth_y

    # Draw QWERTY keyboard
    over_key = False
    selected_key = None
    for row_idx, row in enumerate(keyboard_rows):
        y1 = keyboard_y + row_idx * (key_height + 10)
        # Center the row
        row_width = len(row) * (key_width + 10) - 10
        x_start = (width - row_width) // 2
        for col_idx, key in enumerate(row):
            x1 = x_start + col_idx * (key_width + 10)
            x2 = x1 + key_width
            y2 = y1 + key_height
            # Highlight if index finger is over this key
            if x1 < ix < x2 and y1 < iy < y2:
                over_key = True
                selected_key = key
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 200), -1)
                cv2.rectangle(image, (x1, y1), (x2, y2), (100, 100, 100), 2)
            # Draw key label
            label = ' ' if key == 'SPACE' else key
            font_scale = 1.2 if len(label) == 1 else 1.0
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = x1 + (key_width - text_size[0]) // 2
            text_y = y1 + (key_height + text_size[1]) // 2
            cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 2)

    # Typing logic
    if selected_key:
        # If tap gesture (thumb and index close)
        distance = np.hypot(ix - tx, iy - ty)
        if distance < 40:
            if not pinch_active:
                pinch_active = True
                # Type the key
                if selected_key == 'SPACE':
                    pyautogui.write(' ')
                    last_typed = ' '
                elif selected_key == 'BACKSPACE':
                    pyautogui.press('backspace')
                    last_typed = '<'
                elif selected_key == 'ENTER':
                    pyautogui.press('enter')
                    last_typed = '\n'
                else:
                    pyautogui.write(selected_key)
                    last_typed = selected_key
        else:
            pinch_active = False
    else:
        # If not over a key, allow pinch to click
        if ix != 0 and iy != 0 and tx != 0 and ty != 0:
            distance = np.hypot(ix - tx, iy - ty)
            if distance < 40:
                if not pinch_active:
                    pyautogui.mouseDown()
                    pinch_active = True
                    cv2.putText(image, "Click/Drag!", (ix, iy-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                if pinch_active:
                    pyautogui.mouseUp()
                    pinch_active = False

    # Show last typed character
    cv2.rectangle(image, (20, 20), (400, 100), (255,255,255), -1)
    cv2.putText(image, f"Last typed: {repr(last_typed)}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)

    # Display the image
    cv2.imshow("QWERTY Virtual Keyboard + Mouse", image)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
