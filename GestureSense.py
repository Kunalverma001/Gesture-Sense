import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the webcam feed
cap = cv2.VideoCapture(0)


# Set the window size dynamically
win_width = 1280   # Customize width
win_height = 720   # Customize height
fullscreen = False    # True for full screen, False for custom size

# Configure the OpenCV window based on user settings
window_name = "GestureSense"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

if fullscreen:
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, win_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, win_height)
    cv2.resizeWindow(window_name, win_width, win_height)

# Variables for scrolling, clicking, minimizing, and pointer control
scroll_delay = 0.05
scroll_up_step = 40   
scroll_down_step = -40 

# Initialize the hand tracking model
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Get screen dimensions for pointer scaling
        screen_width, screen_height = pyautogui.size()

        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands and landmarks
        result = hands.process(rgb_frame)

        if result.multi_handedness and result.multi_hand_landmarks:
            for landmarks, hand_type in zip(result.multi_hand_landmarks, result.multi_handedness):
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the hand is left or right
                is_left_hand = hand_type.classification[0].label == 'Left'

                # Left hand for pointer control
                if is_left_hand:
                    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    # Convert normalized coordinates to screen pixel values for pointer
                    pointer_x = int(index_tip.x * screen_width)
                    pointer_y = int(index_tip.y * screen_height)

                    # Move mouse pointer
                    pyautogui.moveTo(pointer_x, pointer_y)

                # Right hand for scrolling, clicking, and minimizing
                else:
                    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                    ring_mcp = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                    # Convert normalized coordinates to pixel values
                    thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                    idx_tip_x, idx_tip_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                    idx_pip_x, idx_pip_y = int(index_pip.x * frame.shape[1]), int(index_pip.y * frame.shape[0])
                    ring_x, ring_y = int(ring_mcp.x * frame.shape[1]), int(ring_mcp.y * frame.shape[0])
                    pinky_x, pinky_y = int(pinky_tip.x * frame.shape[1]), int(pinky_tip.y * frame.shape[0])

                    # Distances for scrolling gestures
                    thumb_idx_tip_dist = math.sqrt((thumb_x - idx_tip_x)**2 + (thumb_y - idx_tip_y)**2)
                    thumb_idx_pip_dist = math.sqrt((thumb_x - idx_pip_x)**2 + (thumb_y - idx_pip_y)**2)

                    # Scroll up when thumb touches index finger tip
                    if thumb_idx_tip_dist < 30:
                        print("Scrolling Up!")
                        pyautogui.scroll(scroll_up_step)
                        time.sleep(scroll_delay)

                    # Scroll down when thumb touches index finger PIP
                    elif thumb_idx_pip_dist < 30:
                        print("Scrolling Down!")
                        pyautogui.scroll(scroll_down_step)
                        time.sleep(scroll_delay)

                    # Click action (thumb touches ring MCP)
                    thumb_ring_dist = math.sqrt((thumb_x - ring_x)**2 + (thumb_y - ring_y)**2)
                    if thumb_ring_dist < 30:
                        print("Click action detected!")
                        pyautogui.click()
                        time.sleep(0.5)

                    # Minimize action (thumb touches pinky tip)
                    thumb_pinky_dist = math.sqrt((thumb_x - pinky_x)**2 + (thumb_y - pinky_y)**2)
                    if thumb_pinky_dist < 30:
                        print("Minimize window detected!")
                        pyautogui.hotkey('alt', 'space')  # Open the window menu
                        pyautogui.press('n')  # Select 'Minimize'
                        time.sleep(0.5)  # Prevent multiple minimizes from a single gesture

        # Show the frame
        cv2.imshow(window_name, frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()