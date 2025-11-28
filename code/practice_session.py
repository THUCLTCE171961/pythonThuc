"""
Custom User Training with Socket Communication
Receives user_id from WinForms and loads appropriate user models
Falls back to general models if user models not found
"""

import argparse
import os
import time
import pickle
import joblib
from collections import deque
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import socket
import struct

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Constants
BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES_TO_PROCESS = 12
MIN_DELTA_MAG = 0.05
RESULT_DISPLAY_SECONDS = 2.0
STATIC_HOLD_SECONDS = 1.0
INSTRUCTION_WINDOW = "Pose Instructions"
DELTA_WEIGHT = 10.0
CONFIDENCE_THRESHOLD = 0.65

# ✅ NEW: Multilingual status messages (same as training_session_ml.py)
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
        "en": "Wrong right fingers: got {}, expected {}",
        "vi": "Ngón tay phải sai: thực tế {}, mong đợi {}"
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
        "en": "Wrong axis: expected {} movement",
        "vi": "Trục sai: mong đợi chuyển động {}"
    },
    "WRONG_DIRECTION": {
        "en": "Wrong direction: expected {}",
        "vi": "Hướng sai: mong đợi {}"
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


# ✅ SOCKET COMMUNICATION FUNCTIONS
def receive_pose_and_user(host='127.0.0.1', port=7000):
    """Receive pose name and user_id from WinForms
    Format: pose_name|user_id
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    print(f"[PYTHON] Waiting for pose and user from WinForms on port {port}...")
    conn, addr = s.accept()
    print(f"[PYTHON] WinForms connected: {addr}")
    
    data = conn.recv(1024).decode().strip()
    conn.sendall(b'OK')
    conn.close()
    s.close()
    
    # Parse format: pose_name|user_id
    if '|' in data:
        pose_name, user_id = data.split('|', 1)
        print(f"[PYTHON] Received - Pose: {pose_name}, User: {user_id}")
        return pose_name, user_id
    else:
        # Fallback for old format (just pose_name)
        print(f"[PYTHON] Received - Pose: {data} (no user_id)")
        return data, None


def start_camera_server(host='127.0.0.1', port=6001):
    """Start camera frame streaming server"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[PYTHON] Camera server listening at {host}:{port}...")
    conn, addr = server.accept()
    print(f"[PYTHON] WinForms camera connected: {addr}")
    return conn


def send_frame_to_winforms(conn, frame):
    """Send camera frame to WinForms"""
    result, img_encoded = cv2.imencode('.jpg', frame)
    if not result:
        print("[PYTHON] Frame encoding failed!")
        return False
    
    data = img_encoded.tobytes()
    try:
        conn.sendall(struct.pack(">L", len(data)) + data)
        return True
    except Exception as e:
        print(f"[PYTHON] Frame send error: {e}")
        return False


def start_status_server(host='127.0.0.1', port=6002):
    """Start training status streaming server"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[PYTHON] Status server listening at {host}:{port}...")
    conn, addr = server.accept()
    print(f"[PYTHON] WinForms status connected: {addr}")
    return conn


def send_status_to_winforms(conn, result, pose, correct, wrong, acc, message_key, *args):
    """Send training status with bilingual messages
    Format: RESULT|POSE|CORRECT|WRONG|ACC|EN_REASON|VI_REASON
    """
    try:
        # Get bilingual messages
        if message_key in STATUS_MESSAGES:
            en_msg = STATUS_MESSAGES[message_key]["en"]
            vi_msg = STATUS_MESSAGES[message_key]["vi"]
            
            # Format messages with arguments if provided
            if args:
                en_msg = en_msg.format(*args)
                vi_msg = vi_msg.format(*args)
        else:
            # Fallback for custom messages
            en_msg = message_key
            vi_msg = message_key
        
        # Send bilingual status
        text = f"{result}|{pose}|{correct}|{wrong}|{acc:.1f}|{en_msg}|{vi_msg}"
        conn.sendall(text.encode('utf-8'))
        return True
    except Exception as e:
        print(f"[PYTHON] Status send error: {e}")
        return False


# ✅ MODEL LOADING FUNCTIONS
def find_user_models(user_id: str):
    """Find user-specific model folder"""
    if not user_id:
        return None
    
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    
    # Search locations for user models
    user_search_locations = [
        script_dir / f'user_{user_id}' / 'models',
        parent_dir / f'user_{user_id}' / 'models',
        Path(f'user_{user_id}') / 'models'
    ]
    
    for models_dir in user_search_locations:
        if models_dir.exists() and (models_dir / 'motion_svm_model.pkl').exists():
            print(f"✅ Found user models: {models_dir}")
            return models_dir
    
    print(f"⚠️  No user models found for user_id: {user_id}")
    return None


def find_general_models():
    """Find general/default model folder"""
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    
    # Search locations for general models
    general_search_locations = [
        script_dir / 'models',
        parent_dir / 'models',
        Path('models')
    ]
    
    for models_dir in general_search_locations:
        if models_dir.exists() and (models_dir / 'motion_svm_model.pkl').exists():
            print(f"✅ Found general models: {models_dir}")
            return models_dir
    
    print(f"❌ No general models found")
    return None


def load_models_smart(user_id: str = None):
    """Smart model loading - try user models first, fallback to general"""
    models_dir = None
    model_type = "general"
    
    # Try user models first if user_id provided
    if user_id:
        models_dir = find_user_models(user_id)
        if models_dir:
            model_type = f"user_{user_id}"
    
    # Fallback to general models
    if models_dir is None:
        models_dir = find_general_models()
        if models_dir is None:
            raise FileNotFoundError("No models found (neither user nor general)")
    
    print(f"📂 Loading models from: {models_dir} (type: {model_type})")
    
    # Load model files
    model_pkl = models_dir / 'motion_svm_model.pkl'
    scaler_pkl = models_dir / 'motion_scaler.pkl'
    static_dynamic_pkl = models_dir / 'static_dynamic_classifier.pkl'
    
    if not model_pkl.exists() or not scaler_pkl.exists():
        raise FileNotFoundError(f"Model files not found in {models_dir}")
    
    try:
        # Try joblib first (new format)
        svm_model = joblib.load(model_pkl)
        scaler = joblib.load(scaler_pkl)
        
        # Load static/dynamic classifier
        static_dynamic_model = None
        if static_dynamic_pkl.exists():
            static_dynamic_model = joblib.load(static_dynamic_pkl)
        
        # Extract label encoder
        if hasattr(svm_model, 'classes_'):
            label_encoder = type('LabelEncoder', (), {
                'classes_': svm_model.classes_,
                'inverse_transform': lambda self, y: svm_model.classes_[y]
            })()
        else:
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(['end', 'home', 'next_slide', 'previous_slide', 
                                              'rotate_down', 'rotate_left', 'rotate_right', 'rotate_up', 
                                              'zoom_in', 'zoom_out'])
        
        print("✅ Models loaded (joblib format)")
        
    except:
        # Fallback to pickle format
        with open(model_pkl, 'rb') as f:
            model_data = pickle.load(f)
        
        with open(scaler_pkl, 'rb') as f:
            scaler = pickle.load(f)
        
        static_dynamic_model = None
        if static_dynamic_pkl.exists():
            with open(static_dynamic_pkl, 'rb') as f:
                static_dynamic_model = pickle.load(f)
        
        svm_model = model_data['model']
        label_encoder = model_data['label_encoder']
        
        print("✅ Models loaded (pickle format)")
    
    print(f"   - Model type: {model_type}")
    print(f"   - Classes: {len(label_encoder.classes_)}")
    print(f"   - Gestures: {list(label_encoder.classes_)}")
    
    return svm_model, label_encoder, scaler, static_dynamic_model, model_type


def load_gesture_templates(user_id: str = None):
    """Load gesture templates - try user templates first, fallback to general"""
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    
    template_file = None
    
    # Try user-specific templates first
    if user_id:
        user_template_locations = [
            script_dir / f"user_{user_id}" / 'training_results' / 'gesture_data_compact.csv',
            parent_dir / f"user_{user_id}" / 'training_results' / 'gesture_data_compact.csv',
            Path(f"user_{user_id}") / 'training_results' / 'gesture_data_compact.csv'
        ]
        
        for user_templates in user_template_locations:
            if user_templates.exists():
                template_file = user_templates
                print(f"✅ Using {user_id}'s personalized gesture templates")
                break
    
    # Fallback to general templates
    if template_file is None:
        general_template_locations = [
            script_dir / 'training_results' / 'gesture_data_compact.csv',
            parent_dir / 'training_results' / 'gesture_data_compact.csv',
            Path('training_results') / 'gesture_data_compact.csv'
        ]
        
        for general_templates in general_template_locations:
            if general_templates.exists():
                template_file = general_templates
                print(f"✅ Using general gesture templates")
                break
    
    if template_file is None or not template_file.exists():
        raise FileNotFoundError("Gesture templates not found")
    
    # Load templates
    import pandas as pd
    df = pd.read_csv(template_file)
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


# ✅ GESTURE DETECTION FUNCTIONS (same as original)
def get_finger_states(hand_landmarks, handedness_label: str) -> List[int]:
    """Extract finger states with improved thumb detection"""
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

    # Palm orientation
    v1 = [mcp_middle.x - wrist.x, mcp_middle.y - wrist.y]
    v2 = [mcp_pinky.x - wrist.x, mcp_pinky.y - wrist.y]
    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
    palm_facing = 1 if cross_z > 0 else -1

    # Thumb detection
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

    # Other fingers
    states[1] = 1 if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y else 0
    states[2] = 1 if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y else 0
    states[3] = 1 if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y else 0
    states[4] = 1 if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y else 0
    return states


def is_fist(hand_landmarks) -> bool:
    """Check if hand is in fist position"""
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
    """Extract wrist position"""
    if not hand_landmarks:
        return None
    wrist = hand_landmarks.landmark[0]
    return np.array([wrist.x, wrist.y], dtype=float)


def smooth_sequence(seq_xy: List[np.ndarray], window: int = 3) -> List[np.ndarray]:
    """Smooth motion sequence"""
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
    """Compute motion features for ML prediction"""
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


def prepare_features(left_states: List[int], right_states: List[int], motion_features: Dict, 
                    scaler, use_expected_left: bool = False, expected_left: List[int] = None) -> np.ndarray:
    """Prepare features - auto-detect format based on scaler"""
    actual_left = expected_left if (use_expected_left and expected_left) else left_states
    
    # Auto-detect feature format
    try:
        if hasattr(scaler, 'n_features_in_'):
            expected_features = scaler.n_features_in_
        elif hasattr(scaler, 'scale_'):
            expected_features = len(scaler.scale_)
        else:
            expected_features = None
    except:
        expected_features = None
    
    if expected_features == 8:
        # General models format
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
        
    elif expected_features == 18:
        # User models format
        weighted_delta_x = motion_features['delta_x'] * DELTA_WEIGHT
        weighted_delta_y = motion_features['delta_y'] * DELTA_WEIGHT
        
        feature_vector = np.array([
            actual_left[0], actual_left[1], actual_left[2], actual_left[3], actual_left[4],
            right_states[0], right_states[1], right_states[2], right_states[3], right_states[4],
            motion_features['main_axis_x'],
            motion_features['main_axis_y'],
            weighted_delta_x,
            weighted_delta_y,
            motion_features['motion_left'] * DELTA_WEIGHT,
            motion_features['motion_right'] * DELTA_WEIGHT,
            motion_features['motion_up'] * DELTA_WEIGHT,
            motion_features['motion_down'] * DELTA_WEIGHT
        ], dtype=float).reshape(1, -1)
        
        X_scaled = scaler.transform(feature_vector)
        return X_scaled
    
    else:
        # Fallback to general format
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


def evaluate_with_ml(left_states: List[int], right_states: List[int], motion_features: Dict, 
                    target_gesture: str, svm_model, label_encoder, scaler, static_dynamic_data, 
                    gesture_templates: Dict, duration: float) -> Tuple[bool, str, List]:
    """Enhanced evaluation - returns (success, message_key, args)"""
    
    if target_gesture not in gesture_templates:
        return False, "NO_TEMPLATE", [target_gesture]
    
    expected = gesture_templates[target_gesture]
    
    print(f"🎯 Target: {target_gesture}")
    print(f"📏 Expected fingers L:{expected['left_fingers']} R:{expected['right_fingers']}")
    print(f"📏 Recorded fingers L:{left_states} R:{right_states}")
    
    # Step 1: Finger validation (only RIGHT hand)
    if right_states != expected['right_fingers']:
        return False, "RIGHT_FINGERS_WRONG", [right_states, expected['right_fingers']]
    
    print("✅ Right hand finger positions correct!")
    
    # Step 2: Static/Dynamic classification
    is_static_expected = expected['is_static']
    
    # Step 3: Static gesture validation
    if is_static_expected:
        print("🏠 Validating static gesture...")
        
        if duration < STATIC_HOLD_SECONDS:
            return False, "STATIC_DURATION_SHORT", [f"{duration:.1f}", f"{STATIC_HOLD_SECONDS}"]
        
        if motion_features['delta_magnitude'] > 0.05:
            return False, "STATIC_TOO_MUCH_MOTION", [f"{motion_features['delta_magnitude']:.3f}"]
        
        print("✅ Static gesture validated!")
        return True, "STATIC_CORRECT", [f"{duration:.1f}"]
    
    # Step 4: Dynamic gesture validation
    print("🔄 Validating dynamic gesture...")
    
    if motion_features['delta_magnitude'] < MIN_DELTA_MAG:
        return False, "MOTION_TOO_SMALL", [f"{motion_features['delta_magnitude']:.3f}"]
    
    # Step 5: Direction validation
    expected_dx = expected['delta_x']
    expected_dy = expected['delta_y']
    actual_dx = motion_features['raw_dx']
    actual_dy = motion_features['raw_dy']
    
    expected_main_x = expected['main_axis_x']
    actual_main_x = motion_features['main_axis_x']
    
    if expected_main_x != actual_main_x:
        axis_name = "horizontal" if expected_main_x else "vertical"
        return False, "WRONG_AXIS", [axis_name]
    
    if expected_main_x == 1:
        if (expected_dx > 0 and actual_dx <= 0) or (expected_dx < 0 and actual_dx >= 0):
            direction = "right" if expected_dx > 0 else "left"
            return False, "WRONG_DIRECTION", [direction]
    else:
        if (expected_dy > 0 and actual_dy <= 0) or (expected_dy < 0 and actual_dy >= 0):
            direction = "down" if expected_dy > 0 else "up"
            return False, "WRONG_DIRECTION", [direction]
    
    print("✅ Direction correct!")
    
    # Step 6: ML confidence validation
    try:
        X = prepare_features(
            left_states, right_states, motion_features, scaler,
            use_expected_left=True, expected_left=expected['left_fingers']
        )
        
        prediction = svm_model.predict(X)[0]
        
        # Get confidence
        try:
            probabilities = svm_model.predict_proba(X)[0]
            confidence = np.max(probabilities)
        except AttributeError:
            try:
                decision_scores = svm_model.decision_function(X)[0]
                if len(decision_scores) > 1:
                    confidence = np.max(decision_scores) / (np.max(decision_scores) - np.min(decision_scores) + 1e-6)
                else:
                    confidence = 1.0 / (1.0 + np.exp(-decision_scores))
                confidence = min(1.0, max(0.5, confidence))
            except:
                confidence = 0.8
        
        predicted_label = prediction if isinstance(prediction, str) else str(prediction)
        
        print(f"🤖 ML Prediction: {predicted_label} (confidence: {confidence:.3f})")
        
        confidence_threshold = CONFIDENCE_THRESHOLD if hasattr(svm_model, 'predict_proba') else 0.5
        if confidence < confidence_threshold:
            return False, "LOW_CONFIDENCE", [f"{confidence:.1%}", f"{confidence_threshold:.0%}"]
        
        if predicted_label != target_gesture:
            return False, "WRONG_PREDICTION", [predicted_label, f"{confidence:.1%}"]
        
        print("✅ ML validation passed!")
        return True, "ML_CORRECT", [f"{confidence:.1%}"]
        
    except Exception as e:
        print(f"❌ ML Evaluation error: {e}")
        return False, "EVALUATION_ERROR", [str(e)]


# ✅ MAIN TRAINING SESSION WITH SOCKET
def run_custom_training_session(camera_index: int = 0,
                                pose_label: str = "",
                                user_id: str = None,
                                cam_conn=None,
                                status_conn=None):
    """Run custom user training session with socket communication"""

    # Load models and templates (smart loading based on user_id)
    try:
        svm_model, label_encoder, scaler, static_dynamic_data, model_type = load_models_smart(user_id)
        gesture_templates = load_gesture_templates(user_id)
        available_gestures = list(label_encoder.classes_)
    except Exception as e:
        print(f"❌ Failed to load models/templates: {e}")
        return

    if not pose_label or pose_label not in gesture_templates:
        print(f"❌ Pose label '{pose_label}' invalid or not found in templates.")
        return

    target_gesture = pose_label
    target_template = gesture_templates[target_gesture]

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    # Training session state
    stats = AttemptStats()
    motion_buffer = deque(maxlen=BUFFER_SIZE)
    state = "IDLE"
    recorded_left_states = None
    recorded_right_states = None
    recording_start_time = None
    status_text = ""
    status_timestamp = 0.0

    def update_status(message: str) -> None:
        nonlocal status_text, status_timestamp
        status_text = message
        status_timestamp = time.time()

    print(f"\n🚀 Custom Training Session Started!")
    print(f"🎯 Target: {target_gesture}")
    print(f"👤 User: {user_id if user_id else 'None (using general models)'}")
    print(f"🤖 Model Type: {model_type}")

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

            # ===== SEND FRAME TO WINFORMS =====
            if cam_conn:
                send_frame_to_winforms(cam_conn, frame)

            # ===== STATE MACHINE =====
            if state == "IDLE":
                if left_confident and left_is_fist:
                    if not right_confident:
                        update_status("❌ Right hand not visible clearly")
                        if status_conn:
                            send_status_to_winforms(
                                status_conn, "WRONG", target_gesture,
                                stats.correct, stats.wrong, stats.accuracy()*100,
                                "RIGHT_HAND_NOT_DETECTED"
                            )
                        continue
                    recorded_left_states = left_states[:]
                    recorded_right_states = right_states[:]
                    motion_buffer.clear()
                    recording_start_time = time.time()
                    state = "RECORDING"
                    update_status("🔴 Recording gesture...")
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
                    stats.record(False, "insufficient_data")
                    update_status("❌ Insufficient recording data")
                    if status_conn:
                        send_status_to_winforms(
                            status_conn, "WRONG", target_gesture,
                            stats.correct, stats.wrong, stats.accuracy()*100,
                            "INSUFFICIENT_DATA"
                        )
                    state = "IDLE"
                    continue
                try:
                    smoothed = smooth_sequence(list(motion_buffer), window=SMOOTHING_WINDOW)
                    motion_features = compute_motion_features(smoothed)
                    if motion_features is None:
                        stats.record(False, "motion_processing_failed")
                        update_status("❌ Failed to process motion")
                        if status_conn:
                            send_status_to_winforms(
                                status_conn, "WRONG", target_gesture,
                                stats.correct, stats.wrong, stats.accuracy()*100,
                                "NO_MOTION_FEATURES"
                            )
                    else:
                        # ML Evaluation with bilingual support
                        success, message_key, args = evaluate_with_ml(
                            recorded_left_states, recorded_right_states, 
                            motion_features, target_gesture, 
                            svm_model, label_encoder, scaler, static_dynamic_data,
                            gesture_templates, duration
                        )

                        reason_display = args[0] if args else ""
                        stats.record(success, reason_display)
                        
                        if status_conn:
                            send_status_to_winforms(
                                status_conn,
                                "CORRECT" if success else "WRONG",
                                target_gesture,
                                stats.correct, stats.wrong, stats.accuracy()*100,
                                message_key, *args
                            )
                        
                        if success:
                            update_status(f"✅ {reason_display}")
                        else:
                            update_status(f"❌ {reason_display}")

                except Exception as e:
                    stats.record(False, f"evaluation_error: {str(e)}")
                    update_status(f"❌ Evaluation error: {str(e)}")
                    if status_conn:
                        send_status_to_winforms(
                            status_conn, "WRONG", target_gesture,
                            stats.correct, stats.wrong, stats.accuracy()*100,
                            "EVALUATION_ERROR", str(e)
                        )
                
                state = "IDLE"
                recorded_left_states = None
                recorded_right_states = None
                recording_start_time = None
                motion_buffer.clear()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Final Statistics:")
        print(f"   Target: {target_gesture}")
        print(f"   User: {user_id if user_id else 'General'}")
        print(f"   Model: {model_type}")
        print(f"   Correct: {stats.correct}  Wrong: {stats.wrong}  Total: {stats.correct + stats.wrong}  Acc: {stats.accuracy()*100:.1f}%")
        print("🏁 Custom Training Session Complete!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Custom user training with socket communication")
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera index for OpenCV (default: 0)')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    print("🤖 Custom User Training Session with Socket Communication")
    print("=" * 60)
    
    # Receive pose name and user_id from WinForms
    pose_label, user_id = receive_pose_and_user()
    if not pose_label:
        print("[WARN] No pose_label received. Exiting.")
        return 1

    # Start socket servers
    cam_conn = start_camera_server()
    status_conn = start_status_server()

    try:
        run_custom_training_session(
            camera_index=args.camera_index,
            pose_label=pose_label,
            user_id=user_id,
            cam_conn=cam_conn,
            status_conn=status_conn
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        try:
            cam_conn.close()
            status_conn.close()
        except:
            pass
    
    print("Custom training session ended.")
    return 0


if __name__ == "__main__":
    exit(main())