import os
import pickle
import collections
import time

import cv2
import mediapipe as mp
import numpy as np
import socket
import struct

import sys
sys.stdout.reconfigure(encoding='utf-8')


def open_camera_socket(host='127.0.0.1', port=6000, retries=3, delay=1.0):
    """Kết nối socket để gửi camera frame đến C# với retry nhẹ"""
    for attempt in range(1, retries + 1):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((host, port))
            print(f"[PYTHON] Connected to C# camera receiver on {host}:{port} (attempt {attempt})")
            return s
        except Exception as e:
            print(f"[PYTHON] Không kết nối được tới C# (attempt {attempt}/{retries}): {e}")
            s.close()
            if attempt < retries:
                time.sleep(delay)
    print("[PYTHON] Bỏ gửi frame vì không kết nối được sau nhiều lần thử.")
    return None


def send_frame_to_winforms(s, frame):
    """Gửi frame qua socket đến C#. Trả về True/False tùy send thành công hay không"""
    if s is None:
        return False

    result, img_encoded = cv2.imencode('.jpg', frame)
    if not result:
        print("Lỗi encode JPEG.")
        return False
    data = img_encoded.tobytes()
    try:
        s.sendall(struct.pack(">L", len(data)) + data)
        return True
    except Exception as e:
        print("Lỗi gửi frame sang C#: ", e)
        return False


def send_command_to_winforms(gesture_name, host='127.0.0.1', port=5006):
    """Gửi tên gesture qua TCP socket tới WinForms server"""
    try:
        print(f"\n[PYTHON] ===== SENDING GESTURE =====")
        print(f"[PYTHON] Gesture name: {gesture_name}")
        print(f"[PYTHON] Connecting to {host}:{port}...")

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))

        print(f"[PYTHON] Connected! Sending: '{gesture_name}'")
        s.sendall(gesture_name.encode('utf-8'))

        print(f"[PYTHON] Command sent successfully!")
        print(f"[PYTHON] ===========================\n")

        s.close()
    except Exception as e:
        print(f"[PYTHON] ERROR sending command: {e}")


# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_parts = BASE_DIR.split(os.sep)
user_folder = None

# Check if we're in a user folder (contains 'user_' in path)
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
MIN_DELTA_MAG = 0.001
DELTA_WEIGHT = 15.0
DISPLAY_DURATION = 3.0
# MIN_PREDICTION_CONFIDENCE = 0.40
MIN_PREDICTION_CONFIDENCE = 0.55

# Static gesture detection
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
mp_drawing = mp.solutions.drawing_utils


def load_gesture_patterns_from_training_data(compact_file=None):
    """Auto-generate gesture patterns from training data với error handling"""
    try:
        import pandas as pd

        if compact_file is None:
            compact_file = os.path.join(TRAINING_RESULTS_DIR, 'gesture_data_compact.csv')

        if not os.path.exists(compact_file):
            print(f"⚠️  Warning: Compact dataset not found: {compact_file}")
            print("   Using fallback patterns...")
            return get_fallback_patterns()

        df = pd.read_csv(compact_file)
        print(f"✅ Loaded compact dataset: {len(df)} samples")

        if 'pose_label' in df.columns:
            gesture_col = 'pose_label'
        elif 'gesture' in df.columns:
            gesture_col = 'gesture'
        else:
            possible_gesture_cols = [col for col in df.columns
                                     if 'gesture' in col.lower() or 'pose' in col.lower()]
            if possible_gesture_cols:
                gesture_col = possible_gesture_cols[0]
                print(f"   Detected gesture column: {gesture_col}")
            else:
                print("⚠️  Warning: No gesture column found, using fallback patterns")
                return get_fallback_patterns()

        finger_cols = [col for col in df.columns if 'finger' in col and 'state' in col]
        if not finger_cols:
            finger_cols = [col for col in df.columns
                           if 'finger' in col or 'thumb' in col or 'index' in col]

        if len(finger_cols) < 5:
            print(f"⚠️  Warning: Only found {len(finger_cols)} finger columns, expected 5")
            return get_fallback_patterns()

        print(f"   Using gesture column: {gesture_col}")
        print(f"   Using finger columns: {finger_cols[:5]}")

        patterns = {}
        unique_gestures = sorted(df[gesture_col].unique())

        for gesture in unique_gestures:
            gesture_data = df[df[gesture_col] == gesture]
            pattern_counts = {}

            for _, row in gesture_data.iterrows():
                try:
                    finger_states = []
                    for i in range(min(5, len(finger_cols))):
                        col = finger_cols[i]
                        state = int(float(row[col])) if pd.notna(row[col]) else 0
                        finger_states.append(state)

                    pattern_tuple = tuple(finger_states)
                    pattern_counts[pattern_tuple] = pattern_counts.get(pattern_tuple, 0) + 1
                except (ValueError, KeyError):
                    continue

            if pattern_counts:
                sorted_patterns = sorted(pattern_counts.items(),
                                         key=lambda x: x[1], reverse=True)
                top_patterns = []
                total_samples = len(gesture_data)

                for pattern, count in sorted_patterns:
                    if count / total_samples > 0.03:
                        top_patterns.append(list(pattern))
                    if len(top_patterns) >= 3:
                        break

                patterns[gesture] = top_patterns
                print(f"   {gesture}: {len(top_patterns)} patterns from {len(gesture_data)} samples")
            else:
                print(f"   {gesture}: No valid patterns found")

        if not patterns:
            print("⚠️  Warning: No patterns loaded, using fallback")
            return get_fallback_patterns()

        print(f"✅ Loaded {len(patterns)} gesture patterns")
        return patterns

    except Exception as e:
        print(f"⚠️  Warning: Could not load patterns from training data: {e}")
        print("   Using fallback patterns...")
        return get_fallback_patterns()


def get_fallback_patterns():
    """Fallback patterns khi không load được từ training data"""
    return {
        'home': [[1, 0, 0, 0, 0]],
        'end': [[0, 0, 0, 0, 1]],
        'next_slide': [[0, 1, 1, 0, 0]],
        'previous_slide': [[0, 1, 1, 0, 0]],
        'rotate_right': [[1, 1, 0, 0, 0]],
        'rotate_left': [[1, 1, 0, 0, 0]],
        'rotate_up': [[1, 1, 0, 0, 0]],
        'rotate_down': [[1, 1, 0, 0, 0]],
        'zoom_in': [[1, 1, 1, 0, 0]],
        'zoom_out': [[1, 1, 1, 0, 0]],
        'zoom_in_slide': [[0, 1, 1, 0, 0]],
        'zoom_out_slide': [[0, 1, 1, 0, 0]],
        'start_present': [[1, 1, 1, 1, 1]],
        'end_present': [[1, 1, 1, 1, 1]],
    }


GESTURE_PATTERNS = load_gesture_patterns_from_training_data()
print(f"Loaded {len(GESTURE_PATTERNS)} gesture patterns from training data")


def load_model():
    """Load trained model, scaler, and static/dynamic classifier với error handling"""
    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(f"Model file not found: {MODEL_PKL}")
    if not os.path.exists(SCALER_PKL):
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PKL}")

    try:
        with open(MODEL_PKL, 'rb') as f:
            model_data = pickle.load(f)

        with open(SCALER_PKL, 'rb') as f:
            scaler = pickle.load(f)

        if 'model' not in model_data or 'label_encoder' not in model_data:
            raise ValueError("Invalid model format: missing 'model' or 'label_encoder'")

        model = model_data['model']
        label_encoder = model_data['label_encoder']

        if not hasattr(scaler, 'transform'):
            raise ValueError("Invalid scaler: missing transform method")

        expected_features = getattr(scaler, 'n_features_in_', None)
        if expected_features:
            print(f"✅ Model expects {expected_features} motion features")
        else:
            print("⚠️  Warning: Could not determine expected features from scaler")

        print(f"✅ Model loaded: {type(model).__name__}")
        print(f"✅ Available gestures: {len(label_encoder.classes_)} - {list(label_encoder.classes_)}")

        static_dynamic_data = None
        if os.path.exists(STATIC_DYNAMIC_PKL):
            try:
                with open(STATIC_DYNAMIC_PKL, 'rb') as f:
                    static_dynamic_data = pickle.load(f)
                print("✅ Static/dynamic classifier loaded")
            except Exception as e:
                print(f"⚠️  Warning: Could not load static/dynamic classifier: {e}")

        return model, label_encoder, scaler, static_dynamic_data

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def calculate_confidence_boost(predicted_gesture, finger_pattern, gesture_patterns):
    """Calculate confidence boost based on pattern matching"""
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


def get_finger_states(hand_landmarks, handedness_label):
    """Extract finger states - same as collect_data_hybrid.py"""
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
    """Check if left hand is in trigger position (flexible fist detection)"""
    if hand_landmarks is None:
        return False

    left_states = get_finger_states(hand_landmarks, 'Left')
    complete_fist = left_states == [0, 0, 0, 0, 0]
    thumb_extended_fist = left_states == [1, 0, 0, 0, 0]
    return complete_fist or thumb_extended_fist


def smooth_points(buffer):
    """Smooth motion points"""
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
    """Compute motion features from smoothed points"""
    if len(smoothed) < 2:
        return None
    start = smoothed[0]
    end = smoothed[-1]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])

    delta_mag = np.sqrt(dx ** 2 + dy ** 2)

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


def prepare_features(left_states, right_states, motion_features, scaler):
    """Prepare features cho prediction với adaptive feature handling"""
    try:
        finger_feats = np.array(left_states + right_states, dtype=float).reshape(1, -1)

        scaler_features = getattr(scaler, 'n_features_in_', 8)

        motion_left = 1.0 if motion_features['delta_x'] < 0 else 0.0
        motion_right = 1.0 if motion_features['delta_x'] > 0 else 0.0
        motion_up = 1.0 if motion_features['delta_y'] < 0 else 0.0
        motion_down = 1.0 if motion_features['delta_y'] > 0 else 0.0

        if scaler_features == 8:
            motion_array = np.array([[
                motion_features['main_axis_x'],
                motion_features['main_axis_y'],
                motion_features['delta_x'] * DELTA_WEIGHT,
                motion_features['delta_y'] * DELTA_WEIGHT,
                motion_left * DELTA_WEIGHT,
                motion_right * DELTA_WEIGHT,
                motion_up * DELTA_WEIGHT,
                motion_down * DELTA_WEIGHT
            ]], dtype=float)

            motion_scaled = scaler.transform(motion_array)
            X = np.hstack([finger_feats, motion_scaled])

        elif scaler_features == 4:
            motion_array = np.array([[
                motion_features['main_axis_x'],
                motion_features['main_axis_y'],
                motion_features['delta_x'] * DELTA_WEIGHT,
                motion_features['delta_y'] * DELTA_WEIGHT
            ]], dtype=float)

            motion_scaled = scaler.transform(motion_array)
            direction_features = np.array([[motion_left, motion_right, motion_up, motion_down]], dtype=float)
            motion_combined = np.hstack([motion_scaled, direction_features])
            X = np.hstack([finger_feats, motion_combined])

        else:
            print(f"⚠️  Warning: Unexpected scaler features {scaler_features}, using basic adaptation")
            motion_array = np.array([[
                motion_features['main_axis_x'],
                motion_features['main_axis_y'],
                motion_features['delta_x'] * DELTA_WEIGHT,
                motion_features['delta_y'] * DELTA_WEIGHT
            ]], dtype=float)
            try:
                motion_scaled = scaler.transform(motion_array)
                X = np.hstack([finger_feats, motion_scaled])
            except Exception:
                X = np.hstack([finger_feats, motion_array])

        expected_total = 10 + scaler_features
        if X.shape[1] != expected_total:
            print(f"⚠️  Warning: Feature mismatch - expected {expected_total}, got {X.shape[1]}")

        return X

    except Exception as e:
        print(f"❌ Error in prepare_features: {e}")
        return np.zeros((1, 18), dtype=float)


def main():
    print('=== GESTURE RECOGNITION TEST ===')

    cam_socket = None  # sẽ tự kết nối trong vòng lặp
    model = None
    label_encoder = None
    scaler = None
    static_dynamic_data = None
    model_loaded = False

    print("\nInstructions:")
    print("  - Put both hands clearly in frame")
    print("  - Close LEFT fist to start recording gesture")
    print("  - Keep RIGHT hand still for static, move lớn cho dynamic")
    print("  - Open LEFT fist to stop and predict\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera! Check:")
        print("  1. Camera is plugged in")
        print("  2. No other app is using the camera")
        print("  3. Camera permissions are granted")
        return

    print("[INFO] Camera opened successfully!")

    test_ok, test_frame = cap.read()
    if not test_ok:
        print("[ERROR] Camera opened but cannot read frame!")
        cap.release()
        return

    print(f"[INFO] Test frame read successfully! Shape: {test_frame.shape}")

    state = 'WAIT'
    buffer = collections.deque(maxlen=BUFFER_SIZE)
    current_left_states = None
    current_right_states = None

    static_start_time = 0
    static_finger_states = None
    is_holding_static = False

    prediction_text = ""
    confidence_text = ""
    top_predictions_text = []
    debug_features_text = []
    prediction_start_time = 0

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print('[ERROR] Cannot read frame from camera.')
                break

            # 🔁 Quản lý kết nối socket tới WinForms
            if cam_socket is None:
                # mỗi vòng lặp thử 1 lần, không spam
                cam_socket = open_camera_socket('127.0.0.1', 6000, retries=1, delay=0.5)

            if cam_socket is not None:
                ok_send = send_frame_to_winforms(cam_socket, frame)
                if not ok_send:
                    try:
                        cam_socket.close()
                    except Exception:
                        pass
                    cam_socket = None  # để vòng sau reconnect

            # Xử lý gesture
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
                    print('\n>>> Recording gesture...')

            elif state == 'RECORD':
                if right_landmarks and right_conf > MIN_CONFIDENCE:
                    wrist = right_landmarks.landmark[0]
                    buffer.append(np.array([wrist.x, wrist.y], dtype=float))

                    current_right_states = get_finger_states(right_landmarks, 'Right')

                    if len(buffer) > 10:
                        recent_points = list(buffer)[-10:]
                        if len(recent_points) >= 2:
                            start_point = recent_points[0]
                            end_point = recent_points[-1]
                            recent_motion = np.sqrt(
                                (end_point[0] - start_point[0]) ** 2 +
                                (end_point[1] - start_point[1]) ** 2
                            )

                            if recent_motion < STATIC_DETECTION_THRESHOLD:
                                if not is_holding_static:
                                    static_start_time = time.time()
                                    static_finger_states = current_right_states.copy()
                                    is_holding_static = True
                                else:
                                    hold_duration = time.time() - static_start_time
                                    states_consistent = (current_right_states == static_finger_states)
                                    if hold_duration >= STATIC_HOLD_TIME and states_consistent:
                                        print(f'>>> Static gesture detected! Held for {hold_duration:.1f}s')
                                        state = 'PREDICT'
                                    elif not states_consistent:
                                        is_holding_static = False
                                        print(f'>>> Finger states changed, resetting static detection')
                            else:
                                is_holding_static = False

                if not left_is_trigger:
                    state = 'PREDICT'

            elif state == 'PREDICT':
                # 🔁 Lazy-load model lần đầu tiên
                if not model_loaded:
                    try:
                        model, label_encoder, scaler, static_dynamic_data = load_model()
                        model_loaded = True
                        print("[INFO] Model loaded successfully (lazy load in PREDICT).")
                    except Exception as e:
                        print(f"[ERROR] Failed to load model: {e}")
                        state = 'WAIT'
                        buffer.clear()
                        current_left_states = None
                        current_right_states = None
                        is_holding_static = False
                        static_finger_states = None
                        continue

                if current_right_states is None:
                    print('[WARN] No right hand finger state -> skipped.')
                else:
                    try:
                        smoothed = smooth_points(list(buffer)) if len(buffer) > 1 else [[0.5, 0.5], [0.5, 0.5]]
                        motion_features = compute_motion_features(smoothed, is_static=True)

                        if motion_features is None:
                            print('[WARN] Could not compute motion features -> skipped.')
                        else:
                            delta_mag = motion_features['delta_magnitude']

                            if is_holding_static and delta_mag < STATIC_DELTA_THRESHOLD:
                                gesture_type = "STATIC"
                                prediction_finger_states = static_finger_states
                                print(f"  -> Detected as STATIC gesture (held still, delta={delta_mag:.4f})")
                            else:
                                gesture_type = "DYNAMIC"
                                prediction_finger_states = current_right_states
                                print(f"  -> Detected as DYNAMIC gesture (motion detected, delta={delta_mag:.4f})")

                            X = prepare_features(current_left_states or [0, 0, 0, 0, 0],
                                                 prediction_finger_states,
                                                 motion_features,
                                                 scaler)

                            prediction = model.predict(X)[0]
                            probabilities = model.predict_proba(X)[0]
                            confidence = np.max(probabilities)

                            gesture_name = label_encoder.inverse_transform([prediction])[0]
                            finger_states = prediction_finger_states

                            boost_applied = False
                            original_confidence = confidence

                            pattern_boost = calculate_confidence_boost(gesture_name, finger_states, GESTURE_PATTERNS)
                            if pattern_boost != 0.0:
                                confidence = max(0.1, min(0.95, confidence + pattern_boost))
                                boost_applied = True

                            top_indices = np.argsort(probabilities)[::-1][:3]
                            top_3_predictions = []
                            for i, idx in enumerate(top_indices):
                                gesture = label_encoder.inverse_transform([idx])[0]
                                prob = probabilities[idx]
                                top_3_predictions.append(f"{i + 1}. {gesture}: {prob:.3f}")

                            feature_debug = []
                            feature_debug.append(f"Left fingers: {current_left_states}")
                            feature_debug.append(f"Right fingers: {prediction_finger_states}")
                            feature_debug.append(
                                f"Delta magnitude: {delta_mag:.4f} "
                                f"({'<' if delta_mag < STATIC_DELTA_THRESHOLD else '>='} {STATIC_DELTA_THRESHOLD})"
                            )
                            feature_debug.append(
                                f"Motion delta: ({motion_features['delta_x']:.3f}, {motion_features['delta_y']:.3f})"
                            )
                            feature_debug.append(f"Was holding static: {is_holding_static}")
                            if boost_applied:
                                feature_debug.append(
                                    f"Confidence {'boosted' if pattern_boost > 0 else 'penalized'}: "
                                    f"{original_confidence:.3f} → {confidence:.3f}")

                            print(f"\n=== {gesture_type} GESTURE PREDICTION ===")
                            print(f"Top prediction: {gesture_name} (confidence: {confidence:.3f})")
                            print("Top 3 predictions:")
                            for pred in top_3_predictions:
                                print(f"  {pred}")
                            print("Debug info:")
                            for feat in feature_debug:
                                print(f"  {feat}")

                            if confidence >= MIN_PREDICTION_CONFIDENCE:
                                prediction_text = f'{gesture_type}: {gesture_name}'
                                confidence_text = f'Confidence: {confidence:.3f}'
                                send_command_to_winforms(gesture_name)
                            else:
                                prediction_text = f'Low Confidence ({gesture_type})'
                                confidence_text = f'Max confidence: {confidence:.3f}'

                            top_predictions_text = top_3_predictions
                            debug_features_text = feature_debug
                            prediction_start_time = time.time()

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
        if cam_socket is not None:
            try:
                cam_socket.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("\nExited gesture recognition.")


if __name__ == '__main__':
    main()
