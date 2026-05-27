import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import pyautogui
import time
import os

# 1. Verify and Load the Trained Ensemble Brain Model
MODEL_PKL_PATH = 'advanced_hand_model.pkl'
if not os.path.exists(MODEL_PKL_PATH):
    # Fallback to absolute path or display help
    print(f"[-] Warning: '{MODEL_PKL_PATH}' not found in current folder.")
    print("[*] Checking parent folder or Documents root...")
    fallback_paths = [
        MODEL_PKL_PATH,
        '../advanced_hand_model.pkl',
        'c:/Users/karakavalasa sumith/Documents/advanced_hand_model.pkl'
    ]
    for path in fallback_paths:
        if os.path.exists(path):
            MODEL_PKL_PATH = path
            break

if not os.path.exists(MODEL_PKL_PATH):
    print("[-] Error: Model file could not be found. Please run train.ipynb first!")
    exit()

model = joblib.load(MODEL_PKL_PATH)
print(f"[+] Loaded Model Brain from: {MODEL_PKL_PATH}")

# 2. Setup Hand Landmarker Config
# Make sure the task asset path matches or defaults properly
task_path = "hand_landmarker.task"
if not os.path.exists(task_path):
    doc_task_path = "c:/Users/karakavalasa sumith/Documents/hand_landmarker.task"
    if os.path.exists(doc_task_path):
        task_path = doc_task_path

base_options = python.BaseOptions(model_asset_path=task_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# --- HYBRID STATE ENGINE VARIABLES ---
system_active = False     # Master activation unlock gate
activation_timer = 0      # Timestamps of activation window open
x_history = []            # Holds frame coordinate data for swipe calculus
history_length = 7        # Queued frames for swipe evaluation
swipe_threshold = 0.23    # Velocity displacement activation threshold
swipe_cooldown = 0        # Visual cooldown buffer

class_names = {0: "FIST (UNLOCKS)", 1: "PEACE SIGN (UNLOCKS)", 2: "OPEN PALM"}

print("\n=== SWIPE MEDIA CONTROLLER ACTIVE ===")
print("Step 1: Present a FIST or PEACE SIGN to awaken the swipe receiver.")
print("Step 2: Swipe your hand LEFT or RIGHT within 2 seconds to trigger media commands.")
print("Press 'q' in the window frame to stop execution.\n")

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Mirror camera stream
        h, w, _ = frame.shape
        current_time = time.time()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(current_time * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        status_color = (0, 0, 255) # Red for Locked state
        status_text = "LOCKED (Inactive)"
        
        # Check if the 2-second activation window has expired
        if system_active and (current_time - activation_timer > 2.0):
            system_active = False
            x_history.clear()
            print("[*] Receiver window closed. Relocking system.")
            
        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            wrist = hand_landmarks[0]
            
            # Draw visual feedback circle on wrist node
            cv2.circle(frame, (int(wrist.x * w), int(wrist.y * h)), 10, (0, 255, 255), -1)
            
            # Extract features relative to the wrist
            live_row = []
            for lm in hand_landmarks:
                live_row.extend([lm.x - wrist.x, lm.y - wrist.y])
            live_features = np.array(live_row).reshape(1, -1)
            
            # Predict static gesture shape
            predicted_label = model.predict(live_features)[0]
            
            # --- HYBRID STATE MACHINE EXECUTOR ---
            if not system_active:
                # If locked, only FIST (0) or PEACE SIGN (1) can trigger activation
                if predicted_label in [0, 1]:
                    system_active = True
                    activation_timer = current_time
                    x_history.clear()
                    print(f"[!] System unlocked via {class_names[predicted_label]} gesture trigger!")
            else:
                # If unlocked, reset the inactivity timer while hand is active
                activation_timer = current_time
                status_text = "ACTIVE - SWIPE GESTURE RECEIVER"
                status_color = (0, 255, 0) # Green for Active state
                
                # Push coordinates into rolling timeline queue
                x_history.append(wrist.x)
                if len(x_history) > history_length:
                    x_history.pop(0)
                
                # Check swipe delta threshold
                if len(x_history) == history_length and swipe_cooldown == 0:
                    total_delta_x = x_history[-1] - x_history[0]
                    
                    if total_delta_x > swipe_threshold:
                        pyautogui.press('nexttrack')
                        print("[ACTION] Swipe Right -> nexttrack triggered")
                        swipe_cooldown = 30
                        system_active = False
                        x_history.clear()
                    elif total_delta_x < -swipe_threshold:
                        pyautogui.press('prevtrack')
                        print("[ACTION] Swipe Left -> prevtrack triggered")
                        swipe_cooldown = 30
                        system_active = False
                        x_history.clear()
        else:
            x_history.clear()

        # Handle UI Debounce cooling ticks
        if swipe_cooldown > 0:
            swipe_cooldown -= 1
            status_text = f"ACTION COOLDOWN [{swipe_cooldown}]"
            status_color = (0, 165, 255)
            
        if system_active and swipe_cooldown == 0:
            time_left = max(0.0, 2.0 - (current_time - activation_timer))
            cv2.putText(frame, f"Window: {time_left:.1f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        # Draw status heads-up display overlays
        cv2.putText(frame, f"STATUS: {status_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.imshow("Swipe Media Controller Hub", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
