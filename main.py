import cv2
import time
import mediapipe as mp
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Hand Connections
# -------------------------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]

# -------------------------------
# MediaPipe Setup
# -------------------------------
model_path = "hand_landmarker.task"
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

# -------------------------------
# PyAutoGUI Setup
# -------------------------------
pyautogui.FAILSAFE = True
last_action_time = 0
cooldown = 0.8

def press_key(key):
    global last_action_time
    current_time = time.time()
    if current_time - last_action_time > cooldown:
        pyautogui.press(key)
        print("Pressed:", key)
        last_action_time = current_time

# -------------------------------
# Camera
# -------------------------------
camera = cv2.VideoCapture(0)
previous_time = 0

# -------------------------------
# Start Detection
# -------------------------------
with vision.HandLandmarker.create_from_options(options) as landmarker:
    while camera.isOpened():
        success, frame = camera.read()
        if not success:
            break

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # Convert image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)

        # Detect hand
        results = landmarker.detect_for_video(mp_image, timestamp)
        gesture = "CENTER"

        if results.hand_landmarks:
            hand = results.hand_landmarks[0]
            
            # Landmarks
            index = hand[8]
            wrist = hand[0]
            middle_knuckle = hand[9]
            
            # --- 1. Tilt Detection (Lane Shifting) ---
            # Compares knuckle to wrist position
            tilt_diff = middle_knuckle.x - wrist.x
            
            if tilt_diff < -0.08:  # Hand tilted left
                gesture = "TILT LEFT"
                press_key("a") # Map to your game's "Left" key
            elif tilt_diff > 0.08:  # Hand tilted right
                gesture = "TILT RIGHT"
                press_key("d") # Map to your game's "Right" key
            
            # --- 2. Position Detection (Jumping/Sliding/Turning) ---
            # Separate 'if' allows these to work alongside or instead of tilting
            elif index.y < 0.30:
                gesture = "JUMP"
                press_key("up")
            elif index.y > 0.70:
                gesture = "SLIDE"
                press_key("down")
            elif index.x < 0.20:
                gesture = "LEFT"
                press_key("left")
            elif index.x > 0.80:
                gesture = "RIGHT"
                press_key("right")
            else:
                gesture = "CENTER"

            # Draw landmarks
            for point in hand:
                px = int(point.x * frame.shape[1])
                py = int(point.y * frame.shape[0])
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

            # Draw connections
            for start, end in HAND_CONNECTIONS:
                x1 = int(hand[start].x * frame.shape[1])
                y1 = int(hand[start].y * frame.shape[0])
                x2 = int(hand[end].x * frame.shape[1])
                y2 = int(hand[end].y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # FPS calculation
        current = time.time()
        fps = 1 / (current - previous_time) if previous_time else 0
        previous_time = current

        cv2.putText(frame, "Gesture: " + gesture, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "FPS: " + str(int(fps)), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Temple Run Hand Controller", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()           