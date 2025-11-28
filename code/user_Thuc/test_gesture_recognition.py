# import os
# import pickle
# import collections
# import time

# import cv2
# import mediapipe as mp
# import numpy as np
# import socket
# import struct

# import sys
# sys.stdout.reconfigure(encoding='utf-8')
# def open_camera_socket(host='127.0.0.1', port=6000):
#     """Kết nối socket để gửi camera frame đến C#"""
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     try:
#         s.connect((host, port))
#         print("Connected to C# camera receiver!")
#     except Exception as e:
#         print("Không kết nối được tới C#: ", e)
#         s = None
#     return s

# def send_frame_to_winforms(s, frame):
#     """Gửi frame qua socket đến C#"""
#     if s is None:
#         return
#     result, img_encoded = cv2.imencode('.jpg', frame)
#     if not result:
#         print("Lỗi encode JPEG.")
#         return
#     data = img_encoded.tobytes()
#     try:
#         s.sendall(struct.pack(">L", len(data)) + data)
#     except Exception as e:
#         pass

# def send_command_to_winforms(gesture_name, host='127.0.0.1', port=5006):
#     """Gửi tên gesture qua TCP socket tới WinForms server"""
#     try:
#         print(f"\n[PYTHON] ===== SENDING GESTURE =====")
#         print(f"[PYTHON] Gesture name: {gesture_name}")
#         print(f"[PYTHON] Connecting to {host}:{port}...")
        
#         s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         s.connect((host, port))
        
#         print(f"[PYTHON] Connected! Sending: '{gesture_name}'")
#         s.sendall(gesture_name.encode('utf-8'))
        
#         print(f"[PYTHON] Command sent successfully!")
#         print(f"[PYTHON] ===========================\n")
        
#         s.close()
#     except Exception as e:
#         print(f"[PYTHON] ERROR sending command: {e}")
# # === CONFIG ===
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODELS_DIR = os.path.join(BASE_DIR, "models")
# MODEL_PKL = os.path.join(MODELS_DIR, 'motion_svm_model.pkl')
# SCALER_PKL = os.path.join(MODELS_DIR, 'motion_scaler.pkl')
# STATIC_DYNAMIC_PKL = os.path.join(MODELS_DIR, 'static_dynamic_classifier.pkl')
# BUFFER_SIZE = 60
# SMOOTHING_WINDOW = 3
# MIN_FRAMES = 12
# MIN_CONFIDENCE = 0.7
# MIN_DELTA_MAG = 0.001  # Lowered for static gestures (match training)
# DELTA_WEIGHT = 15.0  # Updated to match training script
# DISPLAY_DURATION = 3.0  # Display prediction for 3 seconds
# MIN_PREDICTION_CONFIDENCE = 0.40  # Adaptive threshold - lower for raw, higher after boost

# # Static gesture detection - Auto-adjust for distance
# STATIC_HOLD_TIME = 1.5  # Hold static gesture for 1.5 seconds
# STATIC_DELTA_THRESHOLD = 0.008  # Further lowered for 2m+ distance
# STATIC_DETECTION_THRESHOLD = 0.005  # Lowered for motion detection during recording

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5,
# )
# mp_drawing = mp.solutions.drawing_utils

# LEFT_COLS = [f'left_finger_state_{i}' for i in range(5)]
# RIGHT_COLS = [f'right_finger_state_{i}' for i in range(5)]
# MOTION_COLS = ['main_axis_x', 'main_axis_y', 'delta_x', 'delta_y']


# def load_model():
#     """Load trained model, scaler, and static/dynamic classifier"""
#     if not os.path.exists(MODEL_PKL) or not os.path.exists(SCALER_PKL):
#         raise FileNotFoundError("Model files not found! Please train the model first.")
    
#     with open(MODEL_PKL, 'rb') as f:
#         model_data = pickle.load(f)
    
#     with open(SCALER_PKL, 'rb') as f:
#         scaler = pickle.load(f)
    
#     # Load static/dynamic classifier
#     static_dynamic_data = None
#     if os.path.exists(STATIC_DYNAMIC_PKL):
#         with open(STATIC_DYNAMIC_PKL, 'rb') as f:
#             static_dynamic_data = pickle.load(f)
    
#     return model_data['model'], model_data['label_encoder'], scaler, static_dynamic_data


# def get_finger_states(hand_landmarks, handedness_label):
#     """Extract finger states - same as collect_data_hybrid.py"""
#     states = [0, 0, 0, 0, 0]
#     if hand_landmarks is None:
#         return states

#     wrist = hand_landmarks.landmark[0]
#     mcp_middle = hand_landmarks.landmark[9]
#     mcp_pinky = hand_landmarks.landmark[17]
#     v1 = [mcp_middle.x - wrist.x, mcp_middle.y - wrist.y]
#     v2 = [mcp_pinky.x - wrist.x, mcp_pinky.y - wrist.y]
#     cross_z = v1[0] * v2[1] - v1[1] * v2[0]
#     palm_facing = 1 if cross_z > 0 else -1

#     if handedness_label == 'Right':
#         if palm_facing > 0:
#             states[0] = 1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0
#         else:
#             states[0] = 1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0
#     else:
#         if palm_facing > 0:
#             states[0] = 1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0
#         else:
#             states[0] = 1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0

#     states[1] = 1 if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y else 0
#     states[2] = 1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0
#     states[3] = 1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0
#     states[4] = 1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0
#     return states


# def is_trigger_closed(hand_landmarks):
#     """Check if left hand is in trigger position (flexible fist detection)
    
#     Accepts both patterns as trigger:
#     - [0,0,0,0,0] : Complete fist
#     - [1,0,0,0,0] : Fist with thumb extended (common detection error)
#     """
#     if hand_landmarks is None:
#         return False
    
#     # Get finger states for left hand
#     left_states = get_finger_states(hand_landmarks, 'Left')
    
#     # Accept both trigger patterns
#     complete_fist = left_states == [0, 0, 0, 0, 0]
#     thumb_extended_fist = left_states == [1, 0, 0, 0, 0]
    
#     return complete_fist or thumb_extended_fist


# def smooth_points(buffer):
#     """Smooth motion points"""
#     if not buffer:
#         return []
#     window = SMOOTHING_WINDOW
#     right_points = [entry for entry in buffer]
#     smoothed = []
#     for idx in range(len(right_points)):
#         start = max(0, idx - window // 2)
#         end = min(len(right_points), idx + window // 2 + 1)
#         segment = right_points[start:end]
#         smoothed.append(np.mean(segment, axis=0))
#     return smoothed


# def compute_motion_features(smoothed, is_static=False):
#     """Compute motion features from smoothed points"""
#     if len(smoothed) < 2:
#         return None
#     start = smoothed[0]
#     mid = smoothed[len(smoothed) // 2]
#     end = smoothed[-1]
#     dx = float(end[0] - start[0])
#     dy = float(end[1] - start[1])
    
#     # Calculate delta magnitude
#     delta_mag = np.sqrt(dx**2 + dy**2)
    
#     # For static gestures, allow very small motion but don't return None
#     if not is_static and delta_mag < MIN_DELTA_MAG:
#         return None
    
#     if abs(dx) >= abs(dy):
#         main_x, main_y = 1, 0
#         delta_x, delta_y = dx, 0.0
#     else:
#         main_x, main_y = 0, 1
#         delta_x, delta_y = 0.0, dy
    
#     # Add direction features (matching training script)
#     motion_left = 1.0 if dx < 0 else 0.0
#     motion_right = 1.0 if dx > 0 else 0.0
#     motion_up = 1.0 if dy < 0 else 0.0
#     motion_down = 1.0 if dy > 0 else 0.0
    
#     return {
#         'main_axis_x': main_x,
#         'main_axis_y': main_y,
#         'delta_x': float(delta_x),
#         'delta_y': float(delta_y),
#         'motion_left': motion_left,
#         'motion_right': motion_right,
#         'motion_up': motion_up,
#         'motion_down': motion_down,
#         'delta_magnitude': delta_mag,
#     }


# def prepare_features(left_states, right_states, motion_features, scaler):
#     """Prepare features for prediction - same preprocessing as training"""
#     # Combine finger states
#     finger_feats = np.array(left_states + right_states, dtype=float).reshape(1, -1)
    
#     # Apply delta weight and add direction features
#     motion_array = np.array([[
#         motion_features['main_axis_x'],
#         motion_features['main_axis_y'], 
#         motion_features['delta_x'] * DELTA_WEIGHT,
#         motion_features['delta_y'] * DELTA_WEIGHT,
#         motion_features['motion_left'] * DELTA_WEIGHT,
#         motion_features['motion_right'] * DELTA_WEIGHT,
#         motion_features['motion_up'] * DELTA_WEIGHT,
#         motion_features['motion_down'] * DELTA_WEIGHT
#     ]], dtype=float)
    
#     # Scale motion features
#     motion_scaled = scaler.transform(motion_array)
    
#     # Combine features
#     X = np.hstack([finger_feats, motion_scaled])
#     return X

# def prepare_static_features(left_states, right_states, delta_magnitude, static_scaler):
#     """Prepare features for static/dynamic classification"""
#     # Combine finger states + delta magnitude (matching training)
#     finger_feats = np.array(left_states + right_states + [delta_magnitude], dtype=float).reshape(1, -1)
    
#     # Scale features
#     X_scaled = static_scaler.transform(finger_feats)
#     return X_scaled

# def is_static_gesture(motion_features, static_dynamic_data):
#     """Determine if gesture is static or dynamic"""
#     if static_dynamic_data is None:
#         # Fallback: use delta magnitude threshold
#         return motion_features['delta_magnitude'] < STATIC_DELTA_THRESHOLD
    
#     # Use trained classifier (implement if needed)
#     return motion_features['delta_magnitude'] < STATIC_DELTA_THRESHOLD


# def main():
#     print('=== GESTURE RECOGNITION TEST ===')
    
#     cam_socket = open_camera_socket('127.0.0.1', 6000)
#     # Load model
#     try:
#         model, label_encoder, scaler, static_dynamic_data = load_model()
#         print(f"[INFO] Model loaded successfully!")
#         print(f"[INFO] Available gestures: {list(label_encoder.classes_)}")
#         if static_dynamic_data:
#             static_gestures = static_dynamic_data.get('static_gestures', [])
#             print(f"[INFO] Static gestures: {static_gestures}")
#         else:
#             print(f"[INFO] Using fallback static detection")
#     except Exception as e:
#         print(f"[ERROR] Failed to load model: {e}")
#         return
    
#     print("\nInstructions:")
#     print("  - Optimal distance: 70-90cm (close) or 2m+ (far)")
#     print("  - Put both hands clearly in frame")
#     print("  - Close LEFT fist to start recording gesture")
#     print("  - For STATIC gestures: Keep RIGHT hand perfectly still for 1.5s")
#     print("  - For DYNAMIC gestures: Move RIGHT hand with LARGE, CLEAR motions")
#     print("  - Open LEFT fist to stop and predict")
#     print("  - Press 'q' to quit\n")

#     cap = cv2.VideoCapture(0)
#     # ✅ THÊM: Kiểm tra camera mở được không
#     if not cap.isOpened():
#         print("[ERROR] Cannot open camera! Check:")
#         print("  1. Camera is plugged in")
#         print("  2. No other app is using the camera")
#         print("  3. Camera permissions are granted")
#         return
    
#     print("[INFO] Camera opened successfully!")
    
#     # ✅ Test đọc frame
#     test_ok, test_frame = cap.read()
#     if not test_ok:
#         print("[ERROR] Camera opened but cannot read frame!")
#         cap.release()
#         return
    
#     print(f"[INFO] Test frame read successfully! Shape: {test_frame.shape}")
    
#     cv2.namedWindow('Gesture Recognition', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Gesture Recognition', 1280, 960)

#     state = 'WAIT'
#     buffer = collections.deque(maxlen=BUFFER_SIZE)
#     current_left_states = None
#     current_right_states = None
    
#     # Static gesture tracking
#     static_start_time = 0
#     static_finger_states = None
#     is_holding_static = False
    
#     # Prediction display variables
#     prediction_text = ""
#     confidence_text = ""
#     top_predictions_text = []
#     debug_features_text = []
#     prediction_start_time = 0

#     try:
#         while cap.isOpened():
#             ok, frame = cap.read()
#             if not ok:
#                 print('[ERROR] Cannot read frame from camera.')
#                 break

#             send_frame_to_winforms(cam_socket, frame)
#             frame = cv2.flip(frame, 1)
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb)

#             left_landmarks = None
#             right_landmarks = None
#             left_conf = 0.0
#             right_conf = 0.0

#             # Process hand detections
#             if results.multi_hand_landmarks:
#                 for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#                     handedness = results.multi_handedness[idx].classification[0]
#                     label = handedness.label
#                     score = handedness.score
#                     if label == 'Left':
#                         left_landmarks = hand_landmarks
#                         left_conf = score
#                     elif label == 'Right':
#                         right_landmarks = hand_landmarks
#                         right_conf = score
#                     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             left_is_trigger = is_trigger_closed(left_landmarks)

#             if state == 'WAIT':
#                 # No text display - clean interface
#                 if left_is_trigger and left_conf > MIN_CONFIDENCE and right_conf > MIN_CONFIDENCE and right_landmarks:
#                     current_left_states = get_finger_states(left_landmarks, 'Left')
#                     current_right_states = get_finger_states(right_landmarks, 'Right') 
#                     buffer.clear()
#                     state = 'RECORD'
#                     print('\n>>> Recording gesture...')

#             elif state == 'RECORD':
#                 # Collect motion data
#                 if right_landmarks and right_conf > MIN_CONFIDENCE:
#                     wrist = right_landmarks.landmark[0]
#                     buffer.append(np.array([wrist.x, wrist.y], dtype=float))
                    
#                     # Update right hand finger states continuously
#                     current_right_states = get_finger_states(right_landmarks, 'Right')
                    
#                     # Check for static gesture (consistent finger states + minimal motion)
#                     if len(buffer) > 10:  # Need some buffer for motion calculation
#                         recent_points = list(buffer)[-10:]  # Last 10 points
#                         if len(recent_points) >= 2:
#                             start_point = recent_points[0]
#                             end_point = recent_points[-1]
#                             recent_motion = np.sqrt((end_point[0] - start_point[0])**2 + 
#                                                   (end_point[1] - start_point[1])**2)
                            
#                             # Check if motion is minimal (static gesture)
#                             if recent_motion < STATIC_DETECTION_THRESHOLD:
#                                 if not is_holding_static:
#                                     # Start static gesture timer
#                                     static_start_time = time.time()
#                                     static_finger_states = current_right_states.copy()
#                                     is_holding_static = True
#                                 else:
#                                     # Check if held long enough and finger states are consistent
#                                     hold_duration = time.time() - static_start_time
#                                     states_consistent = (current_right_states == static_finger_states)
                                    
#                                     if hold_duration >= STATIC_HOLD_TIME and states_consistent:
#                                         print(f'>>> Static gesture detected! Held for {hold_duration:.1f}s')
#                                         state = 'PREDICT'
#                             else:
#                                 # Motion detected, reset static tracking
#                                 is_holding_static = False
                
#                 # Check if left trigger is opened (no longer in trigger position)
#                 if not left_is_trigger:
#                     # Always go to unified prediction - let delta magnitude decide
#                     state = 'PREDICT'

#             elif state == 'PREDICT':
#                 # Unified prediction for both static and dynamic gestures
#                 final_right_states = static_finger_states if (is_holding_static and static_finger_states) else current_right_states
                
#                 if final_right_states is None:
#                     print('[WARN] No right hand finger state -> skipped.')
#                 else:
#                     try:
#                         # Always compute motion features (handle both static and dynamic)
#                         smoothed = smooth_points(list(buffer)) if len(buffer) > 1 else [[0.5, 0.5], [0.5, 0.5]]
#                         motion_features = compute_motion_features(smoothed, is_static=True)  # Allow small motions
                        
#                         if motion_features is None:
#                             print('[WARN] Could not compute motion features -> skipped.')
#                         else:
#                             # Determine gesture type based on delta magnitude
#                             delta_mag = motion_features['delta_magnitude']
#                             gesture_type = "STATIC" if delta_mag < STATIC_DELTA_THRESHOLD else "DYNAMIC"
                            
#                             # Use static finger states if we detected static gesture, otherwise current states
#                             prediction_finger_states = static_finger_states if (gesture_type == "STATIC" and static_finger_states) else current_right_states
                            
#                             # Prepare features for prediction (match training preprocessing)
#                             X = prepare_features(current_left_states or [0, 0, 0, 0, 0],
#                                                prediction_finger_states or final_right_states,
#                                                motion_features,
#                                                scaler)
                            
#                             # Predict
#                             prediction = model.predict(X)[0]
#                             probabilities = model.predict_proba(X)[0]
#                             confidence = np.max(probabilities)
                            
#                             # Boost confidence for clear patterns
#                             gesture_name = label_encoder.inverse_transform([prediction])[0]
#                             finger_states = prediction_finger_states or final_right_states
                            
#                             # Comprehensive confidence boost for ALL clear patterns
#                             dx, dy = motion_features['delta_x'], motion_features['delta_y']
                            
#                             boost_applied = False
#                             original_confidence = confidence
                            
#                             # Static gestures - high confidence if truly static
#                             if gesture_name in ['home', 'end'] and delta_mag < STATIC_DELTA_THRESHOLD:
#                                 if ((gesture_name == 'home' and finger_states == [1,0,0,0,0]) or
#                                     (gesture_name == 'end' and finger_states == [0,0,0,0,1])):
#                                     confidence = min(0.98, confidence * 1.8)
#                                     boost_applied = True
                            
#                             # Slide gestures - boost for correct finger + direction
#                             elif gesture_name in ['next_slide', 'previous_slide'] and finger_states == [0,1,1,0,0]:
#                                 if ((gesture_name == 'next_slide' and dx > 0.08) or
#                                     (gesture_name == 'previous_slide' and dx < -0.08)):
#                                     confidence = min(0.95, confidence * 1.6)
#                                     boost_applied = True
                            
#                             # Rotate gestures - boost for correct finger + direction
#                             elif gesture_name in ['rotate_right', 'rotate_left', 'rotate_up', 'rotate_down'] and finger_states == [1,1,0,0,0]:
#                                 if ((gesture_name == 'rotate_right' and dx > 0.08) or
#                                     (gesture_name == 'rotate_left' and dx < -0.08) or
#                                     (gesture_name == 'rotate_up' and dy < -0.08) or
#                                     (gesture_name == 'rotate_down' and dy > 0.08)):
#                                     confidence = min(0.95, confidence * 1.6)
#                                     boost_applied = True
                            
#                             # Zoom gestures - boost for correct finger + direction  
#                             elif gesture_name in ['zoom_in', 'zoom_out'] and finger_states == [1,1,1,0,0]:
#                                 if ((gesture_name == 'zoom_in' and dy < -0.08) or
#                                     (gesture_name == 'zoom_out' and dy > 0.08)):
#                                     confidence = min(0.95, confidence * 1.6)
#                                     boost_applied = True
                            
#                             # Get top-3 predictions
#                             top_indices = np.argsort(probabilities)[::-1][:3]
#                             top_3_predictions = []
#                             for i, idx in enumerate(top_indices):
#                                 gesture = label_encoder.inverse_transform([idx])[0]
#                                 prob = probabilities[idx]
#                                 top_3_predictions.append(f"{i+1}. {gesture}: {prob:.3f}")
                            
#                             # Debug: Display feature values
#                             feature_debug = []
#                             feature_debug.append(f"Left fingers: {current_left_states}")
#                             feature_debug.append(f"Right fingers: {prediction_finger_states or final_right_states}")
#                             feature_debug.append(f"Delta magnitude: {delta_mag:.4f} ({'<' if delta_mag < STATIC_DELTA_THRESHOLD else '>='} {STATIC_DELTA_THRESHOLD})")
#                             feature_debug.append(f"Motion delta: ({motion_features['delta_x']:.3f}, {motion_features['delta_y']:.3f})")
#                             feature_debug.append(f"Was holding static: {is_holding_static}")
#                             if boost_applied:
#                                 feature_debug.append(f"Confidence boosted: {original_confidence:.3f} → {confidence:.3f}")
                            
#                             # Get gesture name (already defined above)
#                             # gesture_name = label_encoder.inverse_transform([prediction])[0]
                            
#                             print(f"\n=== {gesture_type} GESTURE PREDICTION ===")
#                             print(f"Top prediction: {gesture_name} (confidence: {confidence:.3f})")
#                             # if confidence >= MIN_PREDICTION_CONFIDENCE:
#                             #     prediction_text = f'{gesture_type}: {gesture_name}'
#                             #     confidence_text = f'Confidence: {confidence:.3f}'
                                
#                             #     # ✅ THÊM DÒNG NÀY ĐỂ GỬI LỆNH ĐẾN C#
#                             #     send_command_to_winforms(gesture_name)
#                             # else:
#                             #     prediction_text = f'Low Confidence ({gesture_type})'
#                             #     confidence_text = f'Max confidence: {confidence:.3f}'
#                             if confidence >= MIN_PREDICTION_CONFIDENCE:
#                                 prediction_text = f'{gesture_type}: {gesture_name}'
#                                 confidence_text = f'Confidence: {confidence:.3f}'
                                
#                                 # ✅ GỬI LỆNH ĐẾN C# CHỈ KHI CONFIDENCE ĐỦ CAO
#                                 send_command_to_winforms(gesture_name)
#                             else:
#                                 prediction_text = f'Low Confidence ({gesture_type})'
#                                 confidence_text = f'Max confidence: {confidence:.3f}'
#                             print("Top 3 predictions:")
#                             for pred in top_3_predictions:
#                                 print(f"  {pred}")
#                             print("Debug info:")
#                             for feat in feature_debug:
#                                 print(f"  {feat}")
                            
#                             # Set display based on confidence threshold
#                             if confidence >= MIN_PREDICTION_CONFIDENCE:
#                                 prediction_text = f'{gesture_type}: {gesture_name}'
#                                 confidence_text = f'Confidence: {confidence:.3f}'
#                             else:
#                                 prediction_text = f'Low Confidence ({gesture_type})'
#                                 confidence_text = f'Max confidence: {confidence:.3f}'
                            
#                             top_predictions_text = top_3_predictions
#                             debug_features_text = feature_debug
#                             prediction_start_time = time.time()
                            
#                     except Exception as e:
#                         print(f'[ERROR] Prediction failed: {e}')

#                 # Reset for next gesture
#                 buffer.clear()
#                 current_left_states = None
#                 current_right_states = None
#                 is_holding_static = False
#                 static_finger_states = None
#                 state = 'WAIT'

#             # Display prediction if within display duration
#             current_time = time.time()
#             if prediction_text and (current_time - prediction_start_time) < DISPLAY_DURATION:
#                 # Main prediction
#                 color = (0, 255, 0) if "Low Confidence" not in prediction_text else (0, 0, 255)
#                 cv2.putText(frame, prediction_text, (20, 50),
#                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
#                 cv2.putText(frame, confidence_text, (20, 100),
#                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
#                 # Top-3 predictions
#                 y_offset = 150
#                 cv2.putText(frame, "Top 3 Predictions:", (20, y_offset),
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#                 for i, pred_text in enumerate(top_predictions_text):
#                     y_offset += 30
#                     cv2.putText(frame, pred_text, (30, y_offset),
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
#                 # Debug features (right side of screen)
#                 debug_x = 700
#                 y_offset = 50
#                 cv2.putText(frame, "Debug Features:", (debug_x, y_offset),
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#                 for feat_text in debug_features_text:
#                     y_offset += 30
#                     cv2.putText(frame, feat_text, (debug_x + 10, y_offset),
#                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
#                 # Add countdown timer
#                 remaining_time = DISPLAY_DURATION - (current_time - prediction_start_time)
#                 timer_text = f'Timer: {remaining_time:.1f}s'
#                 cv2.putText(frame, timer_text, (20, 350),
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

#             cv2.imshow('Gesture Recognition', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         # if cam_socket:
#         #     cam_socket.close()
#         # print("\nExited gesture recognition.")


# if __name__ == '__main__':
#     main()


import os
import pickle
import collections
import time
import cv2
import mediapipe as mp
import numpy as np
import socket
import struct

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Socket config for WinForms (change if needed)
CAMERA_HOST = '127.0.0.1'
CAMERA_PORT = 6000       # WinForms camera receiver port
COMMAND_HOST = '127.0.0.1'
COMMAND_PORT = 5006      # WinForms command port

# Auto-detect user folder based on current path
path_parts = BASE_DIR.split(os.sep)
user_folder = None
for part in path_parts:
    if part.startswith('user_'):
        user_folder = part
        break

if user_folder:
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    TRAINING_RESULTS_DIR = os.path.join(BASE_DIR, "training_results")
    print(f"🔄 Using {user_folder}'s personal model and training data")
else:
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    TRAINING_RESULTS_DIR = os.path.join(BASE_DIR, "training_results")
    print("🔄 Using main model and training data")

MODEL_PKL = os.path.join(MODELS_DIR, 'motion_svm_model.pkl')
SCALER_PKL = os.path.join(MODELS_DIR, 'motion_scaler.pkl')
STATIC_DYNAMIC_PKL = os.path.join(MODELS_DIR, 'static_dynamic_classifier.pkl')
BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES = 12
MIN_CONFIDENCE = 0.7
MIN_DELTA_MAG = 0.0005
DELTA_WEIGHT = 10.0
DISPLAY_DURATION = 3.0
MIN_PREDICTION_CONFIDENCE = 0.60

# === Socket functions ===
def open_camera_socket(host=CAMERA_HOST, port=CAMERA_PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        print(f"Connected to C# camera receiver on {host}:{port}")
    except Exception as e:
        print(f"Không kết nối được tới C#: {e}")
        s = None
    return s

def send_frame_to_winforms(s, frame, resize=True, target_size=(640, 480)):
    """Resize (nếu cần) và gửi frame JPEG qua socket cho WinForms."""
    if s is None:
        return
    try:
        if resize and (frame.shape[1] > target_size[0] or frame.shape[0] > target_size[1]):
            frame_send = cv2.resize(frame, target_size)
        else:
            frame_send = frame
        result, img_encoded = cv2.imencode('.jpg', frame_send)
        if not result:
            print("Lỗi encode JPEG.")
            return
        data = img_encoded.tobytes()
        s.sendall(struct.pack(">L", len(data)) + data)
    except Exception as e:
        print(f"Lỗi gửi frame: {e}")

def send_command_to_winforms(gesture_name, host=COMMAND_HOST, port=COMMAND_PORT):
    """Gửi tên gesture qua TCP socket tới WinForms server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.sendall(gesture_name.encode('utf-8'))
        s.close()
    except Exception as e:
        print(f"[PYTHON] ERROR sending command: {e}")

# === Machine learning & gesture functions ===
def load_gesture_patterns_from_training_data(compact_file=None):
    try:
        import pandas as pd
        if compact_file is None:
            compact_file = os.path.join(TRAINING_RESULTS_DIR, 'gesture_data_compact.csv')
        if not os.path.exists(compact_file):
            print(f"⚠️  Compact dataset not found: {compact_file}")
            return get_fallback_patterns()
        df = pd.read_csv(compact_file)
        if 'pose_label' in df.columns:
            gesture_col = 'pose_label'
        elif 'gesture' in df.columns:
            gesture_col = 'gesture'
        else:
            possible_gesture_cols = [col for col in df.columns if 'gesture' in col.lower() or 'pose' in col.lower()]
            if possible_gesture_cols:
                gesture_col = possible_gesture_cols[0]
            else:
                return get_fallback_patterns()
        finger_cols = [col for col in df.columns if 'finger' in col and 'state' in col]
        if not finger_cols:
            finger_cols = [col for col in df.columns if 'finger' in col or 'thumb' in col or 'index' in col]
        if len(finger_cols) < 5:
            return get_fallback_patterns()
        patterns = {}
        unique_gestures = sorted(df[gesture_col].unique())
        for gesture in unique_gestures:
            gesture_data = df[df[gesture_col] == gesture]
            pattern_counts = {}
            for _, row in gesture_data.iterrows():
                try:
                    finger_states = [int(float(row[finger_cols[i]])) if pd.notna(row[finger_cols[i]]) else 0 for i in range(min(5, len(finger_cols)))]
                    pattern_tuple = tuple(finger_states)
                    pattern_counts[pattern_tuple] = pattern_counts.get(pattern_tuple, 0) + 1
                except (ValueError, KeyError):
                    continue
            if pattern_counts:
                sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
                top_patterns = []
                total_samples = len(gesture_data)
                for pattern, count in sorted_patterns:
                    if count / total_samples > 0.03:
                        top_patterns.append(list(pattern))
                    if len(top_patterns) >= 3:
                        break
                patterns[gesture] = top_patterns
        if not patterns:
            return get_fallback_patterns()
        return patterns
    except Exception as e:
        print(f"⚠️  Could not load patterns from training data: {e}")
        return get_fallback_patterns()

def get_fallback_patterns():
    return {
        'home': [[1,0,0,0,0]],
        'end': [[0,0,0,0,1]],
        'next_slide': [[0,1,1,0,0]],
        'previous_slide': [[0,1,1,0,0]],
        'rotate_right': [[1,1,0,0,0]],
        'rotate_left': [[1,1,0,0,0]],
        'rotate_up': [[1,1,0,0,0]],
        'rotate_down': [[1,1,0,0,0]],
        'zoom_in': [[1,1,1,0,0]],
        'zoom_out': [[1,1,1,0,0]],
    }

STATIC_HOLD_TIME = 1.0
STATIC_DELTA_THRESHOLD = 0.003
STATIC_DETECTION_THRESHOLD = 0.002

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

def load_model():
    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(f"Model file not found: {MODEL_PKL}")
    if not os.path.exists(SCALER_PKL):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PKL}")
    with open(MODEL_PKL, 'rb') as f:
        model_data = pickle.load(f)
    with open(SCALER_PKL, 'rb') as f:
        scaler = pickle.load(f)
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    static_dynamic_data = None
    if os.path.exists(STATIC_DYNAMIC_PKL):
        try:
            with open(STATIC_DYNAMIC_PKL, 'rb') as f:
                static_dynamic_data = pickle.load(f)
        except:
            static_dynamic_data = None
    return model, label_encoder, scaler, static_dynamic_data

GESTURE_PATTERNS = load_gesture_patterns_from_training_data()

def calculate_confidence_boost(predicted_gesture, finger_pattern, gesture_patterns):
    if predicted_gesture not in gesture_patterns:
        return 0.0
    current_pattern = tuple(finger_pattern)
    matching_patterns = gesture_patterns[predicted_gesture]
    if current_pattern in [tuple(p) for p in matching_patterns]:
        return 0.2
    else:
        max_similarity = 0
        for training_pattern in matching_patterns:
            similarity = sum(a == b for a, b in zip(current_pattern, training_pattern)) / 5.0
            max_similarity = max(max_similarity, similarity)
        if max_similarity >= 0.8:
            return 0.1
        elif max_similarity >= 0.6:
            return 0.05
        else:
            return -0.1

def prepare_features(left_finger_states, right_finger_states, motion_features, scaler):
    finger_features = left_finger_states + right_finger_states
    main_x = motion_features['main_axis_x']
    main_y = motion_features['main_axis_y']
    delta_x = motion_features['delta_x'] * 10.0
    delta_y = motion_features['delta_y'] * 10.0
    motion_left = 1.0 if motion_features['delta_x'] < 0 else 0.0
    motion_right = 1.0 if motion_features['delta_x'] > 0 else 0.0
    motion_up = 1.0 if motion_features['delta_y'] < 0 else 0.0
    motion_down = 1.0 if motion_features['delta_y'] > 0 else 0.0
    motion_vals = [main_x, main_y, delta_x, delta_y, motion_left, motion_right, motion_up, motion_down]
    finger_array = np.array([finger_features], dtype=float)
    motion_array = np.array([motion_vals], dtype=float)
    motion_scaled = scaler.transform(motion_array)
    X = np.hstack([finger_array, motion_scaled])
    return X

def get_finger_states(hand_landmarks, handedness_label):
    states = [0, 0, 0, 0, 0]
    if hand_landmarks is None:
        return states
    wrist = hand_landmarks.landmark[0]
    mcp_middle = hand_landmarks.landmark[9]
    mcp_pinky = hand_landmarks.landmark[17]
    v1 = [mcp_middle.x - wrist.x, mcp_middle.y - wrist.y]
    v2 = [mcp_pinky.x - wrist.x, mcp_pinky.y - wrist.y]
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    palm_facing = 1 if cross_z > 0 else -1
    if handedness_label == 'Right':
        if palm_facing > 0:
            states[0] = 1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0
        else:
            states[0] = 1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0
    else:
        if palm_facing > 0:
            states[0] = 1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0
        else:
            states[0] = 1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0
    states[1] = 1 if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y else 0
    states[2] = 1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0
    states[3] = 1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0
    states[4] = 1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0
    return states

def is_trigger_closed(hand_landmarks):
    if hand_landmarks is None:
        return False
    left_states = get_finger_states(hand_landmarks, 'Left')
    complete_fist = left_states == [0, 0, 0, 0, 0]
    thumb_extended_fist = left_states == [1, 0, 0, 0, 0]
    return complete_fist or thumb_extended_fist

def smooth_points(buffer):
    if not buffer:
        return []
    window = SMOOTHING_WINDOW
    right_points = [entry for entry in buffer]
    smoothed = []
    for idx in range(len(right_points)):
        start = max(0, idx - window // 2)
        end = min(len(right_points), idx + window // 2 + 1)
        segment = right_points[start:end]
        smoothed.append(np.mean(segment, axis=0))
    return smoothed

def compute_motion_features(smoothed, is_static=False):
    if len(smoothed) < 2:
        return None
    start = smoothed[0]
    end = smoothed[-1]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    delta_mag = np.sqrt(dx**2 + dy**2)
    if not is_static and delta_mag < MIN_DELTA_MAG:
        return None
    if abs(dx) >= abs(dy):
        main_x, main_y = 1, 0
        delta_x, delta_y = dx, 0.0
    else:
        main_x, main_y = 0, 1
        delta_x, delta_y = 0.0, dy
    motion_left = 1.0 if dx < 0 else 0.0
    motion_right = 1.0 if dx > 0 else 0.0
    motion_up = 1.0 if dy < 0 else 0.0
    motion_down = 1.0 if dy > 0 else 0.0
    return {
        'main_axis_x': main_x,
        'main_axis_y': main_y,
        'delta_x': float(delta_x),
        'delta_y': float(delta_y),
        'motion_left': motion_left,
        'motion_right': motion_right,
        'motion_up': motion_up,
        'motion_down': motion_down,
        'delta_magnitude': delta_mag,
    }

def main():
    print('=== GESTURE RECOGNITION TEST SOCKET MODE ===')
    cam_socket = open_camera_socket()
    try:
        model, label_encoder, scaler, static_dynamic_data = load_model()
        print(f"[INFO] Model loaded successfully! Gestures: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    cap = cv2.VideoCapture(0)

    state = 'WAIT'
    buffer = collections.deque(maxlen=BUFFER_SIZE)
    current_left_states = None
    current_right_states = None
    static_start_time = 0
    static_finger_states = None
    is_holding_static = False

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print('[ERROR] Cannot read frame from camera.')
                break

            # Gửi frame (đã resize 640x480) tới WinForms mỗi vòng lặp
            send_frame_to_winforms(cam_socket, frame, resize=True, target_size=(640, 480))

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left_landmarks = None
            right_landmarks = None
            left_conf = 0.0
            right_conf = 0.0

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0]
                    label = handedness.label
                    score = handedness.score
                    if label == 'Left':
                        left_landmarks = hand_landmarks
                        left_conf = score
                    elif label == 'Right':
                        right_landmarks = hand_landmarks
                        right_conf = score

            left_is_trigger = is_trigger_closed(left_landmarks)

            if state == 'WAIT':
                if left_is_trigger and left_conf > MIN_CONFIDENCE and right_conf > MIN_CONFIDENCE and right_landmarks:
                    current_left_states = get_finger_states(left_landmarks, 'Left')
                    current_right_states = get_finger_states(right_landmarks, 'Right')
                    buffer.clear()
                    state = 'RECORD'

            elif state == 'RECORD':
                if right_landmarks and right_conf > MIN_CONFIDENCE:
                    wrist = right_landmarks.landmark[0]
                    buffer.append(np.array([wrist.x, wrist.y], dtype=float))
                    current_right_states = get_finger_states(right_landmarks, 'Right')
                    if len(buffer) > 5:
                        recent_points = list(buffer)[-5:]
                        if len(recent_points) >= 2:
                            start_point = recent_points[0]
                            end_point = recent_points[-1]
                            recent_motion = np.sqrt((end_point[0] - start_point[0])**2 +
                                                   (end_point[1] - start_point[1])**2)
                            if recent_motion < STATIC_DETECTION_THRESHOLD:
                                if not is_holding_static:
                                    static_start_time = time.time()
                                    static_finger_states = current_right_states.copy()
                                    is_holding_static = True
                                else:
                                    hold_duration = time.time() - static_start_time
                                    states_consistent = (current_right_states == static_finger_states)
                                    if hold_duration >= STATIC_HOLD_TIME and states_consistent:
                                        state = 'PREDICT'
                                    elif not states_consistent:
                                        is_holding_static = False
                            else:
                                is_holding_static = False
                if not left_is_trigger:
                    state = 'PREDICT'

            elif state == 'PREDICT':
                if current_right_states is not None:
                    try:
                        smoothed = smooth_points(list(buffer)) if len(buffer) > 1 else [[0.5, 0.5], [0.5, 0.5]]
                        motion_features = compute_motion_features(smoothed, is_static=True)
                        if motion_features:
                            delta_mag = motion_features['delta_magnitude']
                            if is_holding_static and delta_mag < STATIC_DELTA_THRESHOLD:
                                gesture_type = "STATIC"
                                prediction_finger_states = static_finger_states
                            else:
                                gesture_type = "DYNAMIC"
                                prediction_finger_states = current_right_states
                            X = prepare_features(current_left_states or [0, 0, 0, 0, 0],
                                                prediction_finger_states,
                                                motion_features,
                                                scaler)
                            prediction = model.predict(X)[0]
                            probabilities = model.predict_proba(X)[0]
                            confidence = np.max(probabilities)
                            gesture_name = label_encoder.inverse_transform([prediction])[0]
                            finger_states = prediction_finger_states
                            pattern_boost = calculate_confidence_boost(gesture_name, finger_states, GESTURE_PATTERNS)
                            if pattern_boost != 0.0:
                                confidence = max(0.1, min(0.95, confidence + pattern_boost))
                            if confidence >= MIN_PREDICTION_CONFIDENCE:
                                send_command_to_winforms(gesture_name)
                    except Exception as e:
                        print(f'[ERROR] Prediction failed: {e}')
                buffer.clear()
                current_left_states = None
                current_right_states = None
                is_holding_static = False
                static_finger_states = None
                state = 'WAIT'

    finally:
        cap.release()
        if cam_socket:
            cam_socket.close()
        print("\nExited gesture recognition Python (socket mode).")

if __name__ == '__main__':
    main()