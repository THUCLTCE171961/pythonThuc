# import argparse
# import os
# import time
# import pickle
# from collections import deque
# from typing import Dict, List, Tuple, Optional
# import cv2
# import mediapipe as mp
# import numpy as np
# import socket
# import struct

# import sys
# sys.stdout.reconfigure(encoding='utf-8')

# STATUS_MESSAGES = {
#     "CORRECT": {
#         "en": "CORRECT",
#         "vi": "ĐÚNG"
#     },
#     "WRONG": {
#         "en": "WRONG", 
#         "vi": "SAI"
#     },
#     "RIGHT_HAND_NOT_DETECTED": {
#         "en": "Right hand not detected",
#         "vi": "Không phát hiện tay phải"
#     },
#     "INSUFFICIENT_DATA": {
#         "en": "Not enough frames",
#         "vi": "Không đủ khung hình"
#     },
#     "NO_MOTION_FEATURES": {
#         "en": "No motion features",
#         "vi": "Không có đặc trưng chuyển động"
#     },
#     "EVALUATION_ERROR": {
#         "en": "Evaluation error: {}",
#         "vi": "Lỗi đánh giá: {}"
#     },
#     "RIGHT_FINGERS_WRONG": {
#         "en": "Wrong right fingers",
#         "vi": "Ngón tay phải sai"
#     },
#     "STATIC_DURATION_SHORT": {
#         "en": "Hold longer: {}s < {}s",
#         "vi": "Giữ lâu hơn: {}s < {}s"
#     },
#     "STATIC_TOO_MUCH_MOTION": {
#         "en": "Too much motion: {}",
#         "vi": "Quá nhiều chuyển động: {}"
#     },
#     "STATIC_CORRECT": {
#         "en": "Static gesture held for {}s",
#         "vi": "Gesture tĩnh giữ trong {}s"
#     },
#     "MOTION_TOO_SMALL": {
#         "en": "Movement too small: {}",
#         "vi": "Chuyển động quá nhỏ: {}"
#     },
#     "WRONG_AXIS": {
#         "en": "Wrong axis",
#         "vi": "Sai trục chuyển động"
#     },
#     "WRONG_DIRECTION": {
#         "en": "Wrong direction",
#         "vi": "Chuyển động sai hướng"
#     },
#     "LOW_CONFIDENCE": {
#         "en": "Too uncertain: {} < {}",
#         "vi": "Không chắc chắn: {} < {}"
#     },
#     "WRONG_PREDICTION": {
#         "en": "ML predicted: {} ({})",
#         "vi": "ML dự đoán: {} ({})"
#     },
#     "ML_CORRECT": {
#         "en": "Perfect! ({} confidence)",
#         "vi": "Hoàn hảo! ({} độ tin cậy)"
#     },
#     "NO_TEMPLATE": {
#         "en": "No template found for {}",
#         "vi": "Không tìm thấy mẫu cho {}"
#     }
# }

# def receive_pose_name(host='127.0.0.1', port=7000):
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.bind((host, port))
#     s.listen(1)
#     print("Waiting for pose name from WinForms...")
#     conn, addr = s.accept()
#     pose_name = conn.recv(1024).decode().strip()
#     conn.sendall(b'OK')
#     conn.close()
#     s.close()
#     print(f"Received pose name: {pose_name}")
#     return pose_name

# def start_camera_server(host='127.0.0.1', port=6001):
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server.bind((host, port))
#     server.listen(1)
#     print(f"[PYTHON] Camera server listening at {host}:{port} ...")
#     conn, addr = server.accept()
#     print(f"[PYTHON] WinForms camera connected: {addr}")
#     return conn

# def send_frame_to_winforms(conn, frame):
#     result, img_encoded = cv2.imencode('.jpg', frame)
#     if not result:
#         print("[PYTHON] Frame encoding failed!")
#         return
#     data = img_encoded.tobytes()
#     try:
#         conn.sendall(struct.pack(">L", len(data)) + data)
#     except Exception as e:
#         print("[PYTHON] Frame send error:", e)

# def start_status_server(host='127.0.0.1', port=6002):
#     server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server.bind((host, port))
#     server.listen(1)
#     print(f"[PYTHON] Status server listening at {host}:{port} ...")
#     conn, addr = server.accept()
#     print(f"[PYTHON] WinForms status connected: {addr}")
#     return conn

# def send_status_to_winforms(conn, result, pose, correct, wrong, acc, message_key, *args):
#     """
#     Send training status with both English and Vietnamese text
#     Format: RESULT|POSE|CORRECT|WRONG|ACC|EN_REASON|VI_REASON
#     """
#     try:
#         if message_key in STATUS_MESSAGES:
#             en_msg = STATUS_MESSAGES[message_key]["en"]
#             vi_msg = STATUS_MESSAGES[message_key]["vi"]
#             if args:
#                 en_msg = en_msg.format(*args)
#                 vi_msg = vi_msg.format(*args)
#         else:
#             en_msg = message_key
#             vi_msg = message_key
#         text = f"{result}|{pose}|{correct}|{wrong}|{acc:.1f}|{en_msg}|{vi_msg}"
#         conn.sendall(text.encode())
#         return True
#     except Exception as e:
#         print("[PYTHON] Status send error:", e)
#         return False

# BUFFER_SIZE = 60
# SMOOTHING_WINDOW = 3
# MIN_FRAMES_TO_PROCESS = 12
# MIN_DELTA_MAG = 0.05
# RESULT_DISPLAY_SECONDS = 2.0
# STATIC_HOLD_SECONDS = 1.0
# INSTRUCTION_WINDOW = "Pose Instructions"
# DELTA_WEIGHT = 10.0
# CONFIDENCE_THRESHOLD = 0.55
# MODELS_DIR = 'models'
# MODEL_PKL = os.path.join(MODELS_DIR, 'motion_svm_model.pkl')
# SCALER_PKL = os.path.join(MODELS_DIR, 'motion_scaler.pkl')
# STATIC_DYNAMIC_PKL = os.path.join(MODELS_DIR, 'static_dynamic_classifier.pkl')
# GESTURE_TEMPLATES_CSV = os.path.join('training_results', 'gesture_data_compact.csv')

# class AttemptStats:
#     def __init__(self) -> None:
#         self.correct = 0
#         self.wrong = 0
#         self.last_result = ""
#         self.last_reason = ""
#         self.last_timestamp = 0.0

#     def record(self, success: bool, reason: str) -> None:
#         if success:
#             self.correct += 1
#             self.last_result = "CORRECT"
#         else:
#             self.wrong += 1
#             self.last_result = "WRONG"
#         self.last_reason = reason
#         self.last_timestamp = time.time()

#     def accuracy(self) -> float:
#         total = self.correct + self.wrong
#         return (self.correct / total) if total else 0.0

#     def reset(self) -> None:
#         self.correct = 0
#         self.wrong = 0
#         self.last_result = ""
#         self.last_reason = ""
#         self.last_timestamp = 0.0

# def load_models():
#     if not os.path.exists(MODEL_PKL) or not os.path.exists(SCALER_PKL):
#         raise FileNotFoundError(f"Model files not found! Please check:\n{MODEL_PKL}\n{SCALER_PKL}")
#     with open(MODEL_PKL, 'rb') as f:
#         model_data = pickle.load(f)
#     with open(SCALER_PKL, 'rb') as f:
#         scaler = pickle.load(f)
#     static_dynamic_data = None
#     if os.path.exists(STATIC_DYNAMIC_PKL):
#         with open(STATIC_DYNAMIC_PKL, 'rb') as f:
#             static_dynamic_data = pickle.load(f)
#     print("✅ Models loaded successfully!")
#     print(f"   - SVM Model: {len(model_data['label_encoder'].classes_)} classes")
#     print(f"   - Classes: {list(model_data['label_encoder'].classes_)}")
#     if static_dynamic_data:
#         print(f"   - Static/Dynamic Classifier: Available")
#     return model_data['model'], model_data['label_encoder'], scaler, static_dynamic_data

# def load_gesture_templates():
#     import pandas as pd
#     if not os.path.exists(GESTURE_TEMPLATES_CSV):
#         raise FileNotFoundError(f"Gesture templates not found: {GESTURE_TEMPLATES_CSV}")
#     df = pd.read_csv(GESTURE_TEMPLATES_CSV)
#     templates = {}
#     for _, row in df.iterrows():
#         gesture = row['pose_label']
#         templates[gesture] = {
#             'left_fingers': [int(row[f'left_finger_state_{i}']) for i in range(5)],
#             'right_fingers': [int(row[f'right_finger_state_{i}']) for i in range(5)],
#             'main_axis_x': int(row['main_axis_x']),
#             'main_axis_y': int(row['main_axis_y']),
#             'delta_x': float(row['delta_x']),
#             'delta_y': float(row['delta_y']),
#             'is_static': abs(float(row['delta_x'])) < 0.02 and abs(float(row['delta_y'])) < 0.02
#         }
#     print(f"✅ Gesture templates loaded: {len(templates)} gestures")
#     return templates

# def get_finger_states(hand_landmarks, handedness_label: str) -> List[int]:
#     states = [0, 0, 0, 0, 0]
#     if not hand_landmarks:
#         return states

#     wrist = hand_landmarks.landmark[0]
#     thumb_tip = hand_landmarks.landmark[4]
#     thumb_ip = hand_landmarks.landmark[3]
#     thumb_mcp = hand_landmarks.landmark[2]
#     index_mcp = hand_landmarks.landmark[5]
#     mcp_middle = hand_landmarks.landmark[9]
#     mcp_pinky = hand_landmarks.landmark[17]

#     v1 = [mcp_middle.x - wrist.x, mcp_middle.y - wrist.y]
#     v2 = [mcp_pinky.x - wrist.x, mcp_pinky.y - wrist.y]
#     cross_z = v1[0] * v2[1] - v1[1] * v2[0]
#     palm_facing = 1 if cross_z > 0 else -1

#     palm_center_x = (index_mcp.x + mcp_pinky.x) / 2
#     palm_center_y = (index_mcp.y + mcp_pinky.y) / 2
#     thumb_to_palm_dist = ((thumb_tip.x - palm_center_x)**2 + (thumb_tip.y - palm_center_y)**2)**0.5

#     thumb_extended_x = abs(thumb_tip.x - thumb_mcp.x) > 0.04
#     thumb_extended_y = abs(thumb_tip.y - thumb_mcp.y) > 0.03

#     if handedness_label == "Right":
#         if palm_facing > 0:
#             thumb_position_open = thumb_tip.x < thumb_ip.x
#         else:
#             thumb_position_open = thumb_tip.x > thumb_ip.x
#     else:
#         if palm_facing > 0:
#             thumb_position_open = thumb_tip.x < thumb_ip.x
#         else:
#             thumb_position_open = thumb_tip.x > thumb_ip.x

#     import math
#     def angle_between_points(p1, p2, p3):
#         v1 = [p1.x - p2.x, p1.y - p2.y]
#         v2 = [p3.x - p2.x, p3.y - p2.y]
#         dot_product = v1[0]*v2[0] + v1[1]*v2[1]
#         mag1 = (v1[0]**2 + v1[1]**2)**0.5
#         mag2 = (v2[0]**2 + v2[1]**2)**0.5
#         if mag1 == 0 or mag2 == 0:
#             return 0
#         cos_angle = dot_product / (mag1 * mag2)
#         cos_angle = max(-1, min(1, cos_angle))
#         return math.degrees(math.acos(cos_angle))
#     thumb_angle = angle_between_points(thumb_mcp, thumb_ip, thumb_tip)
#     thumb_straight = thumb_angle > 140

#     distance_open = thumb_to_palm_dist > 0.08
#     extension_open = thumb_extended_x or thumb_extended_y
#     angle_open = thumb_straight
#     thumb_is_open = (distance_open or extension_open or angle_open) and thumb_position_open
#     states[0] = 1 if thumb_is_open else 0

#     states[1] = 1 if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y else 0
#     states[2] = 1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0
#     states[3] = 1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0
#     states[4] = 1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0
#     return states

# def is_fist(hand_landmarks) -> bool:
#     if not hand_landmarks:
#         return False
#     bent = 0
#     if hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y:
#         bent += 1
#     if hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y:
#         bent += 1
#     if hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y:
#         bent += 1
#     if hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y:
#         bent += 1
#     return bent >= 3

# def extract_wrist(hand_landmarks):
#     if not hand_landmarks:
#         return None
#     wrist = hand_landmarks.landmark[0]
#     return np.array([wrist.x, wrist.y], dtype=float)

# def smooth_sequence(seq_xy: List[np.ndarray], window: int = 3) -> List[np.ndarray]:
#     if not seq_xy:
#         return []
#     output = []
#     pad = window // 2
#     for idx in range(len(seq_xy)):
#         start = max(0, idx - pad)
#         end = min(len(seq_xy), idx + pad + 1)
#         chunk = np.array(seq_xy[start:end], dtype=float)
#         output.append(np.mean(chunk, axis=0))
#     return output

# def compute_motion_features(smoothed_xy: List[np.ndarray]) -> Optional[Dict]:
#     n = len(smoothed_xy)
#     if n < 2:
#         return None
#     start = smoothed_xy[0]
#     end = smoothed_xy[-1]
#     dx = float(end[0] - start[0])
#     dy = float(end[1] - start[1])
#     delta_mag = float(np.sqrt(dx*dx + dy*dy))
#     if abs(dx) >= abs(dy):
#         main_x, main_y = 1, 0
#         delta_x, delta_y = dx, 0.0
#     else:
#         main_x, main_y = 0, 1
#         delta_x, delta_y = 0.0, dy
#     motion_left = 1.0 if dx < 0 else 0.0
#     motion_right = 1.0 if dx > 0 else 0.0
#     motion_up = 1.0 if dy < 0 else 0.0
#     motion_down = 1.0 if dy > 0 else 0.0
#     return {
#         'main_axis_x': main_x,
#         'main_axis_y': main_y,
#         'delta_x': float(delta_x),
#         'delta_y': float(delta_y),
#         'raw_dx': dx,
#         'raw_dy': dy,
#         'delta_magnitude': delta_mag,
#         'motion_left': motion_left,
#         'motion_right': motion_right,
#         'motion_up': motion_up,
#         'motion_down': motion_down
#     }

# def prepare_features(left_states: List[int], right_states: List[int], motion_features: Dict, scaler, use_expected_left: bool = False, expected_left: List[int] = None) -> np.ndarray:
#     actual_left = expected_left if (use_expected_left and expected_left) else left_states
#     finger_feats = np.array(actual_left + right_states, dtype=float).reshape(1, -1)
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
#     motion_scaled = scaler.transform(motion_array)
#     X = np.hstack([finger_feats, motion_scaled])
#     return X

# def prepare_static_features(left_states: List[int], right_states: List[int], delta_magnitude: float, static_scaler=None, use_expected_left: bool = False, expected_left: List[int] = None):
#     actual_left = expected_left if (use_expected_left and expected_left) else left_states
#     features = np.array([actual_left + right_states + [delta_magnitude]], dtype=float)
#     if static_scaler:
#         features = static_scaler.transform(features)
#     return features

# def evaluate_with_ml(left_states: List[int], right_states: List[int], motion_features: Dict, 
#                     target_gesture: str, svm_model, label_encoder, scaler, static_dynamic_data, 
#                     gesture_templates: Dict, duration: float) -> Tuple[bool, str, tuple]:
#     if target_gesture not in gesture_templates:
#         return False, "NO_TEMPLATE", (target_gesture, )
#     expected = gesture_templates[target_gesture]
#     if right_states != expected['right_fingers']:
#         return False, "RIGHT_FINGERS_WRONG", (right_states, expected['right_fingers'])
#     is_static_expected = expected['is_static']
#     if static_dynamic_data and 'model' in static_dynamic_data:
#         try:
#             static_features = prepare_static_features(
#                 left_states, right_states, motion_features['delta_magnitude'],
#                 use_expected_left=True, expected_left=expected['left_fingers']
#             )
#             is_static_predicted = static_dynamic_data['model'].predict(static_features)[0] == 'static'
#         except Exception:
#             is_static_predicted = is_static_expected
#     else:
#         is_static_predicted = is_static_expected
#     if is_static_expected:
#         if duration < STATIC_HOLD_SECONDS:
#             return False, "STATIC_DURATION_SHORT", (duration, STATIC_HOLD_SECONDS)
#         if motion_features['delta_magnitude'] > 0.05:
#             return False, "STATIC_TOO_MUCH_MOTION", (motion_features['delta_magnitude'], )
#         return True, "STATIC_CORRECT", (duration, )
#     if motion_features['delta_magnitude'] < MIN_DELTA_MAG:
#         return False, "MOTION_TOO_SMALL", (motion_features['delta_magnitude'], )
#     expected_dx = expected['delta_x']
#     expected_dy = expected['delta_y']
#     actual_dx = motion_features['raw_dx']
#     actual_dy = motion_features['raw_dy']
#     expected_main_x = expected['main_axis_x']
#     actual_main_x = motion_features['main_axis_x']
#     if expected_main_x != actual_main_x:
#         axis_name = "horizontal" if expected_main_x else "vertical"
#         return False, "WRONG_AXIS", (axis_name, )
#     if expected_main_x == 1:
#         if (expected_dx > 0 and actual_dx <= 0) or (expected_dx < 0 and actual_dx >= 0):
#             direction = "right" if expected_dx > 0 else "left"
#             return False, "WRONG_DIRECTION", (direction, )
#     else:
#         if (expected_dy > 0 and actual_dy <= 0) or (expected_dy < 0 and actual_dy >= 0):
#             direction = "down" if expected_dy > 0 else "up"
#             return False, "WRONG_DIRECTION", (direction, )
#     try:
#         X = prepare_features(
#             left_states, right_states, motion_features, scaler,
#             use_expected_left=True, expected_left=expected['left_fingers']
#         )
#         prediction = svm_model.predict(X)[0]
#         probabilities = svm_model.predict_proba(X)[0]
#         confidence = np.max(probabilities)
#         predicted_label = label_encoder.inverse_transform([prediction])[0]
#         if confidence < CONFIDENCE_THRESHOLD:
#             return False, "LOW_CONFIDENCE", (confidence, CONFIDENCE_THRESHOLD)
#         if predicted_label != target_gesture:
#             return False, "WRONG_PREDICTION", (predicted_label, confidence)
#         return True, "ML_CORRECT", (confidence, )
#     except Exception as e:
#         return False, "EVALUATION_ERROR", (str(e), )

# def run_ml_training_session(camera_index: int = 0, pose_label: str = "", cam_conn=None, status_conn=None):
#     try:
#         svm_model, label_encoder, scaler, static_dynamic_data = load_models()
#         gesture_templates = load_gesture_templates()
#     except Exception as e:
#         print(f"❌ Failed to load models/templates: {e}")
#         return
#     if not pose_label or (pose_label not in gesture_templates):
#         print("❌ Pose label invalid or not found in templates.")
#         return
#     target_gesture = pose_label
#     target_template = gesture_templates[target_gesture]
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=2,
#         min_detection_confidence=0.7,
#         min_tracking_confidence=0.5,
#     )
#     mp_drawing = mp.solutions.drawing_utils
#     cap = cv2.VideoCapture(camera_index)
#     if not cap.isOpened():
#         print(f"❌ Could not open camera {camera_index}")
#         return
#     stats = AttemptStats()
#     motion_buffer = deque(maxlen=BUFFER_SIZE)
#     state = "IDLE"
#     recorded_left_states = None
#     recorded_right_states = None
#     recording_start_time = None
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("❌ Camera frame not available")
#                 break
#             frame = cv2.flip(frame, 1)
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = hands.process(rgb)
#             left_landmarks = None
#             right_landmarks = None
#             left_score = 0.0
#             right_score = 0.0
#             if results.multi_hand_landmarks:
#                 for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
#                     handedness = results.multi_handedness[i].classification[0].label
#                     score = results.multi_handedness[i].classification[0].score
#                     if handedness == "Left":
#                         left_landmarks = hand_landmarks
#                         left_score = score
#                     else:
#                         right_landmarks = hand_landmarks
#                         right_score = score
#                     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             left_states = get_finger_states(left_landmarks, "Left") if left_landmarks else [0, 0, 0, 0, 0]
#             right_states = get_finger_states(right_landmarks, "Right") if right_landmarks else [0, 0, 0, 0, 0]
#             left_confident = left_score > 0.6
#             right_confident = right_score > 0.6
#             left_is_fist = is_fist(left_landmarks) if left_landmarks else False
#             if cam_conn:
#                 send_frame_to_winforms(cam_conn, frame)
#             if state == "IDLE":
#                 if left_confident and left_is_fist:
#                     if not right_confident:
#                         if status_conn:
#                             send_status_to_winforms(
#                                 status_conn, "WRONG", target_gesture,
#                                 stats.correct, stats.wrong, stats.accuracy() * 100,
#                                 "RIGHT_HAND_NOT_DETECTED"
#                             )
#                         continue
#                     recorded_left_states = left_states[:]
#                     recorded_right_states = right_states[:]
#                     motion_buffer.clear()
#                     recording_start_time = time.time()
#                     state = "RECORDING"
#                     stats.last_result = ""
#                     stats.last_reason = ""
#                     stats.last_timestamp = 0.0
#             elif state == "RECORDING":
#                 if right_confident:
#                     wrist_pos = extract_wrist(right_landmarks)
#                     if wrist_pos is not None:
#                         motion_buffer.append(wrist_pos)
#                 if left_confident and not left_is_fist:
#                     state = "PROCESSING"
#             elif state == "PROCESSING":
#                 duration = (time.time() - recording_start_time) if recording_start_time else 0.0
#                 if (len(motion_buffer) < MIN_FRAMES_TO_PROCESS or 
#                     recorded_left_states is None or recorded_right_states is None):
#                     stats.record(False, "INSUFFICIENT_DATA")
#                     if status_conn:
#                         send_status_to_winforms(
#                             status_conn, "WRONG", target_gesture,
#                             stats.correct, stats.wrong, stats.accuracy() * 100,
#                             "INSUFFICIENT_DATA"
#                         )
#                     state = "IDLE"
#                     continue
#                 try:
#                     smoothed = smooth_sequence(list(motion_buffer), window=SMOOTHING_WINDOW)
#                     motion_features = compute_motion_features(smoothed)
#                     if motion_features is None:
#                         stats.record(False, "NO_MOTION_FEATURES")
#                         if status_conn:
#                             send_status_to_winforms(
#                                 status_conn, "WRONG", target_gesture,
#                                 stats.correct, stats.wrong, stats.accuracy() * 100,
#                                 "NO_MOTION_FEATURES"
#                             )
#                     else:
#                         success, reason_code, reason_args = evaluate_with_ml(
#                             recorded_left_states, recorded_right_states, 
#                             motion_features, target_gesture, 
#                             svm_model, label_encoder, scaler, static_dynamic_data,
#                             gesture_templates, duration
#                         )
#                         stats.record(success, reason_code)
#                         if status_conn:
#                             send_status_to_winforms(
#                                 status_conn,
#                                 "CORRECT" if success else "WRONG",
#                                 target_gesture,
#                                 stats.correct, stats.wrong, stats.accuracy() * 100,
#                                 reason_code, *reason_args
#                             )
#                 except Exception as e:
#                     stats.record(False, "EVALUATION_ERROR")
#                     if status_conn:
#                         send_status_to_winforms(
#                             status_conn, "WRONG", target_gesture,
#                             stats.correct, stats.wrong, stats.accuracy() * 100,
#                             "EVALUATION_ERROR", str(e)
#                         )
#                 state = "IDLE"
#                 recorded_left_states = None
#                 recorded_right_states = None
#                 recording_start_time = None
#                 motion_buffer.clear()
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         print(f"\n📊 Final Statistics:")
#         print(f"   Target: {target_gesture}")
#         print(f"   Correct: {stats.correct}  Wrong: {stats.wrong}  Total: {stats.correct + stats.wrong}  Acc: {stats.accuracy() * 100:.1f}%")
#         print("🏁 ML Training Session Complete!")

# def parse_args():
#     parser = argparse.ArgumentParser(description="ML-based gesture training session")
#     parser.add_argument('--camera-index', type=int, default=0,
#                        help='Camera index for OpenCV (default: 0)')
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     print("🤖 ML-Based Gesture Training Session")
#     print("=" * 40)
#     pose_label = receive_pose_name()
#     if not pose_label:
#         print("[WARN] No pose_label received. Exiting.")
#         return 1
#     cam_conn = start_camera_server()
#     status_conn = start_status_server()
#     try:
#         run_ml_training_session(
#             camera_index=args.camera_index,
#             pose_label=pose_label,
#             cam_conn=cam_conn,
#             status_conn=status_conn
#         )
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         return 1
#     cam_conn.close()
#     status_conn.close()
#     print("Training session ended.")
#     return 0

# if __name__ == "__main__":
#     exit(main())

import argparse
import os
import time
import pickle
from collections import deque
from typing import Dict, List, Tuple, Optional
import cv2
import mediapipe as mp
import numpy as np
import socket
import struct
import sys

sys.stdout.reconfigure(encoding='utf-8')

STATUS_MESSAGES = {
    "CORRECT": {
        "en": "CORRECT",
        "vi": "ĐÚNG"
    },
    "WRONG": {
        "en": "WRONG", 
        "vi": "SAI"
    },
    "RIGHT_HAND_NOT_DETECTED": {
        "en": "Right hand not detected",
        "vi": "Không phát hiện tay phải"
    },
    "INSUFFICIENT_DATA": {
        "en": "Not enough frames",
        "vi": "Không đủ khung hình"
    },
    "NO_MOTION_FEATURES": {
        "en": "No motion features",
        "vi": "Không có đặc trưng chuyển động"
    },
    "EVALUATION_ERROR": {
        "en": "Evaluation error: {}",
        "vi": "Lỗi đánh giá: {}"
    },
    "RIGHT_FINGERS_WRONG": {
        "en": "Wrong right fingers",
        "vi": "Ngón tay phải sai"
    },
    "STATIC_DURATION_SHORT": {
        "en": "Hold longer: {}s < {}s",
        "vi": "Giữ lâu hơn: {}s < {}s"
    },
    "STATIC_TOO_MUCH_MOTION": {
        "en": "Too much motion: {}",
        "vi": "Quá nhiều chuyển động: {}"
    },
    "STATIC_CORRECT": {
        "en": "Static gesture held for {}s",
        "vi": "Gesture tĩnh giữ trong {}s"
    },
    "MOTION_TOO_SMALL": {
        "en": "Movement too small: {}",
        "vi": "Chuyển động quá nhỏ: {}"
    },
    "WRONG_AXIS": {
        "en": "Wrong axis",
        "vi": "Sai trục chuyển động"
    },
    "WRONG_DIRECTION": {
        "en": "Wrong direction",
        "vi": "Chuyển động sai hướng"
    },
    "LOW_CONFIDENCE": {
        "en": "Too uncertain: {} < {}",
        "vi": "Không chắc chắn: {} < {}"
    },
    "WRONG_PREDICTION": {
        "en": "ML predicted: {} ({})",
        "vi": "ML dự đoán: {} ({})"
    },
    "ML_CORRECT": {
        "en": "Perfect! ({} confidence)",
        "vi": "Hoàn hảo! ({} độ tin cậy)"
    },
    "NO_TEMPLATE": {
        "en": "No template found for {}",
        "vi": "Không tìm thấy mẫu cho {}"
    }
}

# ==== Socket helpers ====

def receive_pose_name(host='127.0.0.1', port=7000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print(f"[PYTHON] Waiting for pose name from WinForms at {host}:{port} ...")
    conn, addr = s.accept()
    print(f"[PYTHON] Pose client connected from {addr}")
    pose_name = conn.recv(1024).decode(errors="ignore").strip()
    conn.sendall(b'OK')
    conn.close()
    s.close()
    print(f"[PYTHON] Received pose name: {pose_name}")
    return pose_name

def start_camera_server(host='127.0.0.1', port=6001):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[PYTHON] Camera server listening at {host}:{port} ...")
    conn, addr = server.accept()
    print(f"[PYTHON] WinForms camera connected: {addr}")
    server.close()
    return conn

def send_frame_to_winforms(conn, frame):
    result, img_encoded = cv2.imencode('.jpg', frame)
    if not result:
        print("[PYTHON] Frame encoding failed!")
        return False
    data = img_encoded.tobytes()
    try:
        conn.sendall(struct.pack(">L", len(data)) + data)
        return True
    except Exception as e:
        print("[PYTHON] Frame send error:", e)
        return False

def start_status_server(host='127.0.0.1', port=6002):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[PYTHON] Status server listening at {host}:{port} ...")
    conn, addr = server.accept()
    print(f"[PYTHON] WinForms status connected: {addr}")
    server.close()
    return conn

def send_status_to_winforms(conn, result, pose, correct, wrong, acc, message_key, *args):
    """
    Send training status with both English and Vietnamese text
    Format: RESULT|POSE|CORRECT|WRONG|ACC|EN_REASON|VI_REASON
    """
    try:
        if message_key in STATUS_MESSAGES:
            en_msg = STATUS_MESSAGES[message_key]["en"]
            vi_msg = STATUS_MESSAGES[message_key]["vi"]
            if args:
                en_msg = en_msg.format(*args)
                vi_msg = vi_msg.format(*args)
        else:
            en_msg = message_key
            vi_msg = message_key
        text = f"{result}|{pose}|{correct}|{wrong}|{acc:.1f}|{en_msg}|{vi_msg}"
        conn.sendall(text.encode("utf-8"))
        return True
    except Exception as e:
        print("[PYTHON] Status send error:", e)
        return False

# ==== Config & model paths ====

BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES_TO_PROCESS = 12
MIN_DELTA_MAG = 0.05
RESULT_DISPLAY_SECONDS = 2.0
STATIC_HOLD_SECONDS = 1.0
DELTA_WEIGHT = 10.0
CONFIDENCE_THRESHOLD = 0.55

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_parts = BASE_DIR.split(os.sep)
user_folder = None
for part in path_parts:
    if part.startswith("user_"):
        user_folder = part
        break

if user_folder:
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    TRAINING_RESULTS_DIR = os.path.join(BASE_DIR, "training_results")
    print(f"🔄 Using {user_folder}'s personal model & training data")
else:
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    TRAINING_RESULTS_DIR = os.path.join(BASE_DIR, "training_results")
    print("🔄 Using main model & training data")

MODEL_PKL = os.path.join(MODELS_DIR, 'motion_svm_model.pkl')
SCALER_PKL = os.path.join(MODELS_DIR, 'motion_scaler.pkl')
STATIC_DYNAMIC_PKL = os.path.join(MODELS_DIR, 'static_dynamic_classifier.pkl')
GESTURE_TEMPLATES_CSV = os.path.join(TRAINING_RESULTS_DIR, 'gesture_data_compact.csv')

# Globals so we only load once
svm_model = None
label_encoder = None
scaler = None
static_dynamic_data = None
gesture_templates = None

class AttemptStats:
    def __init__(self) -> None:
        self.correct = 0
        self.wrong = 0
        self.last_result = ""
        self.last_reason = ""
        self.last_timestamp = 0.0

    def record(self, success: bool, reason: str) -> None:
        if success:
            self.correct += 1
            self.last_result = "CORRECT"
        else:
            self.wrong += 1
            self.last_result = "WRONG"
        self.last_reason = reason
        self.last_timestamp = time.time()

    def accuracy(self) -> float:
        total = self.correct + self.wrong
        return (self.correct / total) if total else 0.0

    def reset(self) -> None:
        self.correct = 0
        self.wrong = 0
        self.last_result = ""
        self.last_reason = ""
        self.last_timestamp = 0.0

# ==== Model & template loading ====

def load_models():
    if not os.path.exists(MODEL_PKL) or not os.path.exists(SCALER_PKL):
        raise FileNotFoundError(f"Model files not found! Please check:\n{MODEL_PKL}\n{SCALER_PKL}")
    with open(MODEL_PKL, 'rb') as f:
        model_data = pickle.load(f)
    with open(SCALER_PKL, 'rb') as f:
        scaler_obj = pickle.load(f)
    static_dynamic = None
    if os.path.exists(STATIC_DYNAMIC_PKL):
        with open(STATIC_DYNAMIC_PKL, 'rb') as f:
            static_dynamic = pickle.load(f)
    print("✅ Models loaded successfully!")
    print(f"   - SVM Model: {len(model_data['label_encoder'].classes_)} classes")
    print(f"   - Classes: {list(model_data['label_encoder'].classes_)}")
    if static_dynamic:
        print("   - Static/Dynamic Classifier: Available")
    return model_data['model'], model_data['label_encoder'], scaler_obj, static_dynamic

def load_gesture_templates():
    import pandas as pd
    if not os.path.exists(GESTURE_TEMPLATES_CSV):
        raise FileNotFoundError(f"Gesture templates not found: {GESTURE_TEMPLATES_CSV}")
    df = pd.read_csv(GESTURE_TEMPLATES_CSV)
    templates = {}
    for _, row in df.iterrows():
        gesture = row['pose_label']
        templates[gesture] = {
            'left_fingers': [int(row[f'left_finger_state_{i}']) for i in range(5)],
            'right_fingers': [int(row[f'right_finger_state_{i}']) for i in range(5)],
            'main_axis_x': int(row['main_axis_x']),
            'main_axis_y': int(row['main_axis_y']),
            'delta_x': float(row['delta_x']),
            'delta_y': float(row['delta_y']),
            'is_static': abs(float(row['delta_x'])) < 0.02 and abs(float(row['delta_y'])) < 0.02
        }
    print(f"✅ Gesture templates loaded: {len(templates)} gestures")
    return templates

def init_models_and_templates():
    global svm_model, label_encoder, scaler, static_dynamic_data, gesture_templates
    try:
        svm_model, label_encoder, scaler, static_dynamic_data = load_models()
        gesture_templates = load_gesture_templates()
        print("[PYTHON] Models & templates ready for training session.")
    except Exception as e:
        print(f"❌ Failed to init models/templates: {e}")
        sys.exit(1)

# ==== Feature extraction helpers ====

def get_finger_states(hand_landmarks, handedness_label: str) -> List[int]:
    states = [0, 0, 0, 0, 0]
    if not hand_landmarks:
        return states

    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_mcp = hand_landmarks.landmark[2]
    index_mcp = hand_landmarks.landmark[5]
    mcp_middle = hand_landmarks.landmark[9]
    mcp_pinky = hand_landmarks.landmark[17]

    v1 = [mcp_middle.x - wrist.x, mcp_middle.y - wrist.y]
    v2 = [mcp_pinky.x - wrist.x, mcp_pinky.y - wrist.y]
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    palm_facing = 1 if cross_z > 0 else -1

    palm_center_x = (index_mcp.x + mcp_pinky.x) / 2
    palm_center_y = (index_mcp.y + mcp_pinky.y) / 2
    thumb_to_palm_dist = ((thumb_tip.x - palm_center_x)**2 + (thumb_tip.y - palm_center_y)**2)**0.5

    thumb_extended_x = abs(thumb_tip.x - thumb_mcp.x) > 0.04
    thumb_extended_y = abs(thumb_tip.y - thumb_mcp.y) > 0.03

    if handedness_label == "Right":
        if palm_facing > 0:
            thumb_position_open = thumb_tip.x < thumb_ip.x
        else:
            thumb_position_open = thumb_tip.x > thumb_ip.x
    else:
        if palm_facing > 0:
            thumb_position_open = thumb_tip.x < thumb_ip.x
        else:
            thumb_position_open = thumb_tip.x > thumb_ip.x

    import math
    def angle_between_points(p1, p2, p3):
        v1 = [p1.x - p2.x, p1.y - p2.y]
        v2 = [p3.x - p2.x, p3.y - p2.y]
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = (v1[0]**2 + v1[1]**2)**0.5
        mag2 = (v2[0]**2 + v2[1]**2)**0.5
        if mag1 == 0 or mag2 == 0:
            return 0
        cos_angle = dot_product / (mag1 * mag2)
        cos_angle = max(-1, min(1, cos_angle))
        return math.degrees(math.acos(cos_angle))
    thumb_angle = angle_between_points(thumb_mcp, thumb_ip, thumb_tip)
    thumb_straight = thumb_angle > 140

    distance_open = thumb_to_palm_dist > 0.08
    extension_open = thumb_extended_x or thumb_extended_y
    angle_open = thumb_straight
    thumb_is_open = (distance_open or extension_open or angle_open) and thumb_position_open
    states[0] = 1 if thumb_is_open else 0

    states[1] = 1 if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y else 0
    states[2] = 1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0
    states[3] = 1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0
    states[4] = 1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0
    return states

def is_fist(hand_landmarks) -> bool:
    if not hand_landmarks:
        return False
    bent = 0
    if hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y:
        bent += 1
    if hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y:
        bent += 1
    if hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y:
        bent += 1
    if hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y:
        bent += 1
    return bent >= 3

def extract_wrist(hand_landmarks):
    if not hand_landmarks:
        return None
    wrist = hand_landmarks.landmark[0]
    return np.array([wrist.x, wrist.y], dtype=float)

def smooth_sequence(seq_xy: List[np.ndarray], window: int = 3) -> List[np.ndarray]:
    if not seq_xy:
        return []
    output = []
    pad = window // 2
    for idx in range(len(seq_xy)):
        start = max(0, idx - pad)
        end = min(len(seq_xy), idx + pad + 1)
        chunk = np.array(seq_xy[start:end], dtype=float)
        output.append(np.mean(chunk, axis=0))
    return output

def compute_motion_features(smoothed_xy: List[np.ndarray]) -> Optional[Dict]:
    n = len(smoothed_xy)
    if n < 2:
        return None
    start = smoothed_xy[0]
    end = smoothed_xy[-1]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    delta_mag = float(np.sqrt(dx*dx + dy*dy))
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
        'raw_dx': dx,
        'raw_dy': dy,
        'delta_magnitude': delta_mag,
        'motion_left': motion_left,
        'motion_right': motion_right,
        'motion_up': motion_up,
        'motion_down': motion_down
    }

def prepare_features(left_states: List[int], right_states: List[int], motion_features: Dict, scaler, use_expected_left: bool = False, expected_left: List[int] = None) -> np.ndarray:
    actual_left = expected_left if (use_expected_left and expected_left) else left_states
    finger_feats = np.array(actual_left + right_states, dtype=float).reshape(1, -1)
    motion_array = np.array([[
        motion_features['main_axis_x'],
        motion_features['main_axis_y'], 
        motion_features['delta_x'] * DELTA_WEIGHT,
        motion_features['delta_y'] * DELTA_WEIGHT,
        motion_features['motion_left'] * DELTA_WEIGHT,
        motion_features['motion_right'] * DELTA_WEIGHT,
        motion_features['motion_up'] * DELTA_WEIGHT,
        motion_features['motion_down'] * DELTA_WEIGHT
    ]], dtype=float)
    motion_scaled = scaler.transform(motion_array)
    X = np.hstack([finger_feats, motion_scaled])
    return X

def prepare_static_features(left_states: List[int], right_states: List[int], delta_magnitude: float, static_scaler=None, use_expected_left: bool = False, expected_left: List[int] = None):
    actual_left = expected_left if (use_expected_left and expected_left) else left_states
    features = np.array([actual_left + right_states + [delta_magnitude]], dtype=float)
    if static_scaler:
        features = static_scaler.transform(features)
    return features

# ==== Evaluation ====

def evaluate_with_ml(left_states: List[int], right_states: List[int], motion_features: Dict, 
                    target_gesture: str, duration: float) -> Tuple[bool, str, tuple]:
    if target_gesture not in gesture_templates:
        return False, "NO_TEMPLATE", (target_gesture, )
    expected = gesture_templates[target_gesture]
    if right_states != expected['right_fingers']:
        return False, "RIGHT_FINGERS_WRONG", (right_states, expected['right_fingers'])
    is_static_expected = expected['is_static']
    if static_dynamic_data and 'model' in static_dynamic_data:
        try:
            static_features = prepare_static_features(
                left_states, right_states, motion_features['delta_magnitude'],
                static_scaler=static_dynamic_data.get('scaler'),
                use_expected_left=True, expected_left=expected['left_fingers']
            )
            is_static_predicted = static_dynamic_data['model'].predict(static_features)[0] == 'static'
        except Exception:
            is_static_predicted = is_static_expected
    else:
        is_static_predicted = is_static_expected  # fallback

    if is_static_expected:
        if duration < STATIC_HOLD_SECONDS:
            return False, "STATIC_DURATION_SHORT", (duration, STATIC_HOLD_SECONDS)
        if motion_features['delta_magnitude'] > 0.05:
            return False, "STATIC_TOO_MUCH_MOTION", (motion_features['delta_magnitude'], )
        return True, "STATIC_CORRECT", (duration, )

    if motion_features['delta_magnitude'] < MIN_DELTA_MAG:
        return False, "MOTION_TOO_SMALL", (motion_features['delta_magnitude'], )

    expected_dx = expected['delta_x']
    expected_dy = expected['delta_y']
    actual_dx = motion_features['raw_dx']
    actual_dy = motion_features['raw_dy']
    expected_main_x = expected['main_axis_x']
    actual_main_x = motion_features['main_axis_x']
    if expected_main_x != actual_main_x:
        axis_name = "horizontal" if expected_main_x else "vertical"
        return False, "WRONG_AXIS", (axis_name, )
    if expected_main_x == 1:
        if (expected_dx > 0 and actual_dx <= 0) or (expected_dx < 0 and actual_dx >= 0):
            direction = "right" if expected_dx > 0 else "left"
            return False, "WRONG_DIRECTION", (direction, )
    else:
        if (expected_dy > 0 and actual_dy <= 0) or (expected_dy < 0 and actual_dy >= 0):
            direction = "down" if expected_dy > 0 else "up"
            return False, "WRONG_DIRECTION", (direction, )

    try:
        X = prepare_features(
            left_states, right_states, motion_features, scaler,
            use_expected_left=True, expected_left=expected['left_fingers']
        )
        prediction = svm_model.predict(X)[0]
        probabilities = svm_model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        if confidence < CONFIDENCE_THRESHOLD:
            return False, "LOW_CONFIDENCE", (confidence, CONFIDENCE_THRESHOLD)
        if predicted_label != target_gesture:
            return False, "WRONG_PREDICTION", (predicted_label, confidence)
        return True, "ML_CORRECT", (confidence, )
    except Exception as e:
        return False, "EVALUATION_ERROR", (str(e), )

# ==== Main training loop ====

def run_ml_training_session(camera_index: int, pose_label: str, cam_conn, status_conn):
    target_gesture = pose_label
    if target_gesture not in gesture_templates:
        print(f"❌ Pose label '{target_gesture}' not found in templates.")
        if status_conn:
            send_status_to_winforms(
                status_conn, "WRONG", target_gesture,
                0, 0, 0.0,
                "NO_TEMPLATE", target_gesture
            )
        return

    print(f"[PYTHON] Starting ML training session for gesture: {target_gesture}")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        if status_conn:
            send_status_to_winforms(
                status_conn, "WRONG", target_gesture,
                0, 0, 0.0,
                "EVALUATION_ERROR", f"Cannot open camera {camera_index}"
            )
        return

    stats = AttemptStats()
    motion_buffer = deque(maxlen=BUFFER_SIZE)
    state = "IDLE"
    recorded_left_states = None
    recorded_right_states = None
    recording_start_time = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Camera frame not available")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            left_landmarks = None
            right_landmarks = None
            left_score = 0.0
            right_score = 0.0

            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[i].classification[0].label
                    score = results.multi_handedness[i].classification[0].score
                    if handedness == "Left":
                        left_landmarks = hand_landmarks
                        left_score = score
                    else:
                        right_landmarks = hand_landmarks
                        right_score = score
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            left_states = get_finger_states(left_landmarks, "Left") if left_landmarks else [0, 0, 0, 0, 0]
            right_states = get_finger_states(right_landmarks, "Right") if right_landmarks else [0, 0, 0, 0, 0]

            left_confident = left_score > 0.6
            right_confident = right_score > 0.6
            left_is_fist = is_fist(left_landmarks) if left_landmarks else False

            if cam_conn:
                ok_send = send_frame_to_winforms(cam_conn, frame)
                if not ok_send:
                    print("[PYTHON] Camera client disconnected, stopping session.")
                    break

            if state == "IDLE":
                if left_confident and left_is_fist:
                    if not right_confident:
                        if status_conn:
                            send_status_to_winforms(
                                status_conn, "WRONG", target_gesture,
                                stats.correct, stats.wrong, stats.accuracy() * 100,
                                "RIGHT_HAND_NOT_DETECTED"
                            )
                        continue
                    recorded_left_states = left_states[:]
                    recorded_right_states = right_states[:]
                    motion_buffer.clear()
                    recording_start_time = time.time()
                    state = "RECORDING"
                    # Giữ lại tổng correct/wrong, chỉ reset result cuối
                    stats.last_result = ""
                    stats.last_reason = ""
                    stats.last_timestamp = 0.0

            elif state == "RECORDING":
                if right_confident:
                    wrist_pos = extract_wrist(right_landmarks)
                    if wrist_pos is not None:
                        motion_buffer.append(wrist_pos)
                if left_confident and not left_is_fist:
                    state = "PROCESSING"

            elif state == "PROCESSING":
                duration = (time.time() - recording_start_time) if recording_start_time else 0.0
                if (len(motion_buffer) < MIN_FRAMES_TO_PROCESS or 
                    recorded_left_states is None or recorded_right_states is None):
                    stats.record(False, "INSUFFICIENT_DATA")
                    if status_conn:
                        send_status_to_winforms(
                            status_conn, "WRONG", target_gesture,
                            stats.correct, stats.wrong, stats.accuracy() * 100,
                            "INSUFFICIENT_DATA"
                        )
                    state = "IDLE"
                    recording_start_time = None
                    motion_buffer.clear()
                    recorded_left_states = None
                    recorded_right_states = None
                    continue
                try:
                    smoothed = smooth_sequence(list(motion_buffer), window=SMOOTHING_WINDOW)
                    motion_features = compute_motion_features(smoothed)
                    if motion_features is None:
                        stats.record(False, "NO_MOTION_FEATURES")
                        if status_conn:
                            send_status_to_winforms(
                                status_conn, "WRONG", target_gesture,
                                stats.correct, stats.wrong, stats.accuracy() * 100,
                                "NO_MOTION_FEATURES"
                            )
                    else:
                        success, reason_code, reason_args = evaluate_with_ml(
                            recorded_left_states, recorded_right_states, 
                            motion_features, target_gesture, duration
                        )
                        stats.record(success, reason_code)
                        if status_conn:
                            send_status_to_winforms(
                                status_conn,
                                "CORRECT" if success else "WRONG",
                                target_gesture,
                                stats.correct, stats.wrong, stats.accuracy() * 100,
                                reason_code, *reason_args
                            )
                except Exception as e:
                    stats.record(False, "EVALUATION_ERROR")
                    if status_conn:
                        send_status_to_winforms(
                            status_conn, "WRONG", target_gesture,
                            stats.correct, stats.wrong, stats.accuracy() * 100,
                            "EVALUATION_ERROR", str(e)
                        )
                state = "IDLE"
                recording_start_time = None
                motion_buffer.clear()
                recorded_left_states = None
                recorded_right_states = None

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Final Statistics:")
        print(f"   Target: {target_gesture}")
        print(f"   Correct: {stats.correct}  Wrong: {stats.wrong}  Total: {stats.correct + stats.wrong}  Acc: {stats.accuracy() * 100:.1f}%")
        print("🏁 ML Training Session Complete!")

# ==== CLI & entrypoint ====

def parse_args():
    parser = argparse.ArgumentParser(description="ML-based gesture training session")
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera index for OpenCV (default: 0)')
    return parser.parse_args()

def main():
    args = parse_args()
    print("🤖 ML-Based Gesture Training Session")
    print("=" * 40)

    # Preload models + templates as soon as script starts
    init_models_and_templates()

    pose_label = receive_pose_name()
    if not pose_label:
        print("[WARN] No pose_label received. Exiting.")
        return 1

    cam_conn = start_camera_server()
    status_conn = start_status_server()

    try:
        run_ml_training_session(
            camera_index=args.camera_index,
            pose_label=pose_label,
            cam_conn=cam_conn,
            status_conn=status_conn
        )
    except Exception as e:
        print(f"❌ Error in training session: {e}")
        return 1
    finally:
        try:
            cam_conn.close()
        except Exception:
            pass
        try:
            status_conn.close()
        except Exception:
            pass

    print("Training session ended.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
