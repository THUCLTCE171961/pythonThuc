import os
import csv
import collections
import sys
import socket
import struct
import signal

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# =========================
#  FIX UNICODE & WORKDIR
# =========================
if os.name == 'nt':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())


def setup_working_directory():
    """Ensure working directory is set to script directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_cwd = os.getcwd()

    print("PYTHON STARTUP DEBUG:")
    print(f"   Script file: {__file__}")
    print(f"   Script directory: {script_dir}")
    print(f"   Current working directory: {current_cwd}")
    print(f"   Arguments: {sys.argv}")

    if current_cwd != script_dir:
        print(f"WARNING: Working directory mismatch! Changing from {current_cwd} to {script_dir}")
        os.chdir(script_dir)
        print(f"SUCCESS: Changed working directory to: {os.getcwd()}")
    else:
        print(f"SUCCESS: Working directory is correct: {script_dir}")

    return script_dir


SCRIPT_DIR = setup_working_directory()

# === CONFIG ===
DEFAULT_CSV = 'training_results/gesture_data_compact.csv'  # Base dataset (read-only)
BUFFER_SIZE = 60
SMOOTHING_WINDOW = 3
MIN_FRAMES = 12
MIN_CONFIDENCE = 0.7

# Quality validation settings
REQUIRED_SAMPLES = 5          # Must collect 5 samples
MIN_CONSISTENT_SAMPLES = 3    # Need at least 3 consistent samples
SIMILARITY_THRESHOLD = 0.85   # Threshold for considering samples "similar"

# Globals
AVAILABLE_GESTURES = []       # (không dùng trong bản WinForms nhưng giữ để tương thích)
COLLECTED_SAMPLES = []        # (để debug nếu cần)
QUALITY_SAMPLES = []
USER_NAME = ""
CUSTOM_CSV = ""

REFERENCE_DATA = None         # Data chuẩn (training_results/gesture_data_compact.csv)
USER_GESTURE_DATA = None      # Data custom của user

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

LEFT_COLUMNS = [f'left_finger_state_{i}' for i in range(5)]
RIGHT_COLUMNS = [f'right_finger_state_{i}' for i in range(5)]
MOTION_COLUMNS = [
    'motion_x_start', 'motion_y_start',
    'motion_x_mid', 'motion_y_mid',
    'motion_x_end', 'motion_y_end',
]
FEATURE_COLUMNS = ['main_axis_x', 'main_axis_y', 'delta_x', 'delta_y']

SESSION_SAMPLES = []  # Samples hợp lệ sau khi validate

# =========================
#  STATUS MESSAGES
# =========================
STATUS_MESSAGES = {
    "START": {
        "en": "Recording started",
        "vi": "Bắt đầu ghi"
    },
    "ERROR_RIGHT_HAND": {
        "en": "Right hand not detected",
        "vi": "Không phát hiện tay phải"
    },
    "ERROR_INSUFFICIENT_DATA": {
        "en": "Insufficient data: {} frames",
        "vi": "Dữ liệu không đủ: {} khung hình"
    },
    "ERROR_MOTION_FEATURES": {
        "en": "Failed to compute motion features",
        "vi": "Không thể tính toán đặc trưng chuyển động"
    },
    "ERROR_LOW_CONFIDENCE": {
        "en": "Sample rejected - low confidence: {:.3f}",
        "vi": "Mẫu bị từ chối - độ tin cậy thấp: {:.3f}"
    },
    "CONFLICT": {
        "en": "CONFLICT: {}",
        "vi": "XUNG ĐỘT: {}"
    },
    "NO_CONFLICT": {
        "en": "OK: No significant conflicts (max score: {:.2f})",
        "vi": "OK: Không có xung đột nghiêm trọng (điểm cao nhất: {:.2f})"
    },
    "SAMPLE_COLLECTED": {
        "en": "Sample collected",
        "vi": "Đã thu thập mẫu"
    },
    "QUALITY_FAILED": {
        "en": "Quality validation failed: {}",
        "vi": "Xác thực chất lượng thất bại: {}"
    },
    "FINISH": {
        "en": "Done",
        "vi": "Hoàn thành"
    },
    "SAVED_FILE": {
        "en": "Saved file: {}",
        "vi": "Đã lưu file: {}"
    }
}


def signal_handler(sig, frame):
    print(f"Received signal: {sig}, terminating...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =========================
#  SOCKET HELPERS (WINFORMS)
# =========================
def is_connection_alive(conn):
    try:
        conn.settimeout(0.1)
        conn.send(b'')
        return True
    except Exception:
        return False
    finally:
        try:
            conn.settimeout(None)
        except:
            pass


def receive_user_and_pose(host='127.0.0.1', port=7000):
    """Nhận 'username|pose_label' từ WinForms"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    print("Waiting for username|pose_label from WinForms...")
    conn, addr = s.accept()
    data = conn.recv(1024).decode().strip()
    conn.sendall(b'OK')
    conn.close()
    s.close()
    try:
        username, pose_label = data.split('|')
    except Exception:
        username = data
        pose_label = ""
    print(f"Received username={username}, pose_label={pose_label}")
    return username, pose_label


def start_camera_server(host='127.0.0.1', port=6001):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"[CAMERA] Listening at {host}:{port}...")
    conn, addr = server.accept()
    print(f"WinForms camera connected: {addr}")
    return conn


def start_status_server(host='127.0.0.1', port=6002):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"[STATUS] Listening at {host}:{port}...")
    conn, addr = server.accept()
    print(f"WinForms status connected: {addr}")
    return conn


def send_frame_to_winforms(conn, frame):
    """Gửi frame JPG cho WinForms (giống bản cũ)"""
    result, img_encoded = cv2.imencode('.jpg', frame)
    if not result:
        print("[ERROR] Frame encoding failed!")
        return False
    data = img_encoded.tobytes()
    try:
        conn.sendall(struct.pack(">L", len(data)) + data)
        return True
    except Exception as e:
        print("[ERROR] Frame send:", e)
        return False


def send_status_to_winforms(conn, event, pose_label, sample_count, conflict, message_key, *args):
    """
    Gửi text status cho WinForms:
    event|pose_label|sample_count|conflict|en_msg|vi_msg
    """
    try:
        if message_key in STATUS_MESSAGES:
            en_msg = STATUS_MESSAGES[message_key]["en"]
            vi_msg = STATUS_MESSAGES[message_key]["vi"]
            if args:
                en_msg = en_msg.format(*args)
                vi_msg = vi_msg.format(*args)
        else:
            # Nếu không có key trong STATUS_MESSAGES -> dùng chính message_key làm nội dung
            en_msg = message_key
            vi_msg = message_key

        text = f"{event}|{pose_label}|{sample_count}|{conflict}|{en_msg}|{vi_msg}"
        conn.sendall(text.encode())
        return True
    except Exception as e:
        print("[ERROR] Status send:", e)
        return False

# =========================
#  QUALITY + CONFLICT LOGIC
# =========================
def calculate_sample_similarity(sample1, sample2):
    """Calculate similarity between two gesture samples"""

    # Compare finger states (most important)
    finger_similarity = 0.0
    finger_keys = ['finger_thumb', 'finger_index', 'finger_middle', 'finger_ring', 'finger_pinky']
    for key in finger_keys:
        if sample1[key] == sample2[key]:
            finger_similarity += 0.2  # Each finger is 20%

    # Compare motion direction (normalized)
    motion1 = np.array([sample1['motion_x'], sample1['motion_y']])
    motion2 = np.array([sample2['motion_x'], sample2['motion_y']])

    # Normalize motion vectors
    norm1 = np.linalg.norm(motion1)
    norm2 = np.linalg.norm(motion2)

    motion_similarity = 0.0
    if norm1 > 0 and norm2 > 0:
        # Calculate cosine similarity
        cosine_sim = np.dot(motion1, motion2) / (norm1 * norm2)
        motion_similarity = (cosine_sim + 1) / 2  # Convert from [-1,1] to [0,1]

    # Weighted combination: finger states (70%) + motion direction (30%)
    total_similarity = finger_similarity * 0.7 + motion_similarity * 0.3

    return total_similarity


def validate_sample_quality(samples_list):
    """
    Validate quality of collected samples
    Returns: (is_valid, consistent_samples, message)
    """

    if len(samples_list) < REQUIRED_SAMPLES:
        return False, [], f"Need {REQUIRED_SAMPLES - len(samples_list)} more samples"

    print(f"\n[SEARCH] QUALITY VALIDATION: Analyzing {len(samples_list)} samples...")

    # Calculate similarity matrix
    similarities = {}
    for i in range(len(samples_list)):
        similarities[i] = {}
        for j in range(len(samples_list)):
            if i != j:
                sim = calculate_sample_similarity(samples_list[i], samples_list[j])
                similarities[i][j] = sim
                print(f"   Sample {i+1} vs {j+1}: {sim:.3f} similarity")
            else:
                similarities[i][j] = 1.0

    # Find groups of consistent samples
    consistent_groups = []
    used_samples = set()

    for i in range(len(samples_list)):
        if i in used_samples:
            continue

        # Start new group with sample i
        group = [i]
        used_samples.add(i)

        # Find other samples similar to sample i
        for j in range(len(samples_list)):
            if j != i and j not in used_samples:
                if similarities[i][j] >= SIMILARITY_THRESHOLD:
                    group.append(j)
                    used_samples.add(j)

        if len(group) >= MIN_CONSISTENT_SAMPLES:
            consistent_groups.append(group)

    print(f"\n[ANALYSIS] CONSISTENCY ANALYSIS:")
    for idx, group in enumerate(consistent_groups):
        group_samples = [samples_list[i] for i in group]
        # Show finger patterns for this group
        finger_patterns = []
        for sample in group_samples:
            fingers = [int(sample[f'finger_{name}']) for name in ['thumb', 'index', 'middle', 'ring', 'pinky']]
            finger_patterns.append(fingers)

        print(f"   Group {idx+1}: {len(group)} samples - Fingers: {finger_patterns[0]}")

        # Show motion directions
        motions = [(sample['motion_x'], sample['motion_y']) for sample in group_samples]
        avg_motion = np.mean(motions, axis=0)
        print(f"           Average motion: ({avg_motion[0]:.3f}, {avg_motion[1]:.3f})")

    if not consistent_groups:
        return False, [], f"No consistent groups found! Need {MIN_CONSISTENT_SAMPLES} similar samples"

    # Use the largest consistent group
    best_group = max(consistent_groups, key=len)
    consistent_samples = [samples_list[i] for i in best_group]

    print(f"\n[OK] VALIDATION RESULT:")
    print(f"   Found {len(consistent_samples)} consistent samples (need {MIN_CONSISTENT_SAMPLES})")
    print(f"   Quality: {len(consistent_samples)}/{len(samples_list)} samples are consistent")

    if len(consistent_samples) >= MIN_CONSISTENT_SAMPLES:
        return True, consistent_samples, f"Quality OK: {len(consistent_samples)} consistent samples"
    else:
        return False, [], f"Need {MIN_CONSISTENT_SAMPLES - len(consistent_samples)} more consistent samples"


def load_reference_data():
    """Load reference data for conflict detection"""
    global REFERENCE_DATA
    if REFERENCE_DATA is None:
        try:
            REFERENCE_DATA = pd.read_csv(DEFAULT_CSV)
            print(f"[OK] Loaded reference data: {len(REFERENCE_DATA)} samples")
        except Exception as e:
            print(f"[ERROR] Failed to load reference data: {e}")
            REFERENCE_DATA = pd.DataFrame()  # Empty fallback
    return REFERENCE_DATA


def load_user_gesture_data(username):
    """Load all existing user gesture data for conflict detection"""
    global USER_GESTURE_DATA
    if USER_GESTURE_DATA is not None:
        return USER_GESTURE_DATA

    user_folder = f"user_{username}"
    raw_data_folder = os.path.join(user_folder, "raw_data")

    if not os.path.exists(raw_data_folder):
        USER_GESTURE_DATA = pd.DataFrame()
        return USER_GESTURE_DATA

    # Load all CSV files in raw_data folder
    all_data = []
    for filename in os.listdir(raw_data_folder):
        if filename.startswith(f"gesture_data_custom_{username}_") and filename.endswith('.csv'):
            try:
                filepath = os.path.join(raw_data_folder, filename)
                df = pd.read_csv(filepath)
                all_data.append(df)
            except Exception as e:
                print(f"[WARNING] Failed to load {filename}: {e}")

    if all_data:
        USER_GESTURE_DATA = pd.concat(all_data, ignore_index=True)
        print(f"[OK] Loaded user data: {len(USER_GESTURE_DATA)} samples from {len(all_data)} files")
    else:
        USER_GESTURE_DATA = pd.DataFrame()
        print(f"[INFO] No existing user data found for {username}")

    return USER_GESTURE_DATA


def get_motion_direction(delta_x, delta_y, threshold=0.01):
    """Get primary motion direction from delta values"""
    abs_x = abs(delta_x)
    abs_y = abs(delta_y)

    if abs_x < threshold and abs_y < threshold:
        return "static"

    if abs_x > abs_y:
        return "right" if delta_x > 0 else "left"
    else:
        return "down" if delta_y > 0 else "up"


def check_gesture_conflict(left_states, right_states, delta_x, delta_y, target_gesture, username=None):
    """
    Check if gesture conflicts with existing reference data and user data
    Returns: (has_conflict, conflict_message)
    """
    # Check against reference data
    ref_data = load_reference_data()
    if not ref_data.empty:
        current_direction = get_motion_direction(delta_x, delta_y)

        for _, row in ref_data.iterrows():
            ref_left = [int(row[f'left_finger_state_{i}']) for i in range(5)]
            ref_right = [int(row[f'right_finger_state_{i}']) for i in range(5)]

            if ref_left == left_states and ref_right == right_states:
                ref_direction = get_motion_direction(row['delta_x'], row['delta_y'])
                ref_gesture = row['pose_label']

                if current_direction == ref_direction:
                    return True, f"CONFLICT with REFERENCE '{ref_gesture}': same pattern + {ref_direction} direction"

    # Check against user data
    if username:
        user_data = load_user_gesture_data(username)
        if not user_data.empty:
            current_direction = get_motion_direction(delta_x, delta_y)

            for _, row in user_data.iterrows():
                user_left = [int(row[f'left_finger_state_{i}']) for i in range(5)]
                user_right = [int(row[f'right_finger_state_{i}']) for i in range(5)]

                if user_left == left_states and user_right == right_states:
                    user_direction = get_motion_direction(row['delta_x'], row['delta_y'])
                    user_gesture = row['pose_label']

                    if current_direction == user_direction:
                        return True, f"CONFLICT with YOUR '{user_gesture}': same pattern + {user_direction} direction"

    return False, f"No conflicts found (direction: {get_motion_direction(delta_x, delta_y)})"


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


def is_fist(hand_landmarks):
    if hand_landmarks is None:
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


def ensure_capture_csv_exists(csv_path):
    if os.path.isfile(csv_path):
        return
    columns = ['instance_id', 'pose_label'] + LEFT_COLUMNS + RIGHT_COLUMNS + MOTION_COLUMNS + FEATURE_COLUMNS
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)


def next_instance_id(csv_path):
    if not os.path.isfile(csv_path):
        return 1
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return 1
        return int(df['instance_id'].max()) + 1
    except Exception:
        return 1


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


def compute_motion_features(smoothed):
    if len(smoothed) < 2:
        return None
    start = smoothed[0]
    mid = smoothed[len(smoothed) // 2]
    end = smoothed[-1]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if abs(dx) >= abs(dy):
        main_x, main_y = 1, 0
        delta_x, delta_y = dx, 0.0
    else:
        main_x, main_y = 0, 1
        delta_x, delta_y = 0.0, dy
    return {
        'start': start,
        'mid': mid,
        'end': end,
        'main_axis_x': main_x,
        'main_axis_y': main_y,
        'delta_x': float(delta_x),
        'delta_y': float(delta_y),
    }


def check_realtime_conflict(new_sample, pose_label, existing_csv=DEFAULT_CSV, username=None):
    """
    Check if new sample conflicts with existing gestures using strict finger+direction matching
    Returns: (has_conflict, conflict_message, conflicting_gestures)
    """
    try:
        left_states = new_sample['left_states']
        right_states = new_sample['right_states']
        features = new_sample['features']
        delta_x = features['delta_x']
        delta_y = features['delta_y']

        has_conflict, conflict_msg = check_gesture_conflict(
            left_states, right_states, delta_x, delta_y, pose_label, username
        )

        if has_conflict:
            return True, conflict_msg, []
        else:
            return False, conflict_msg, []

    except Exception as e:
        return False, f"Error checking conflicts: {e}", []


def save_session_to_user_folder(pose_label):
    """Save all SESSION_SAMPLES vào CSV trong thư mục user_{USER_NAME}"""
    if not SESSION_SAMPLES:
        return None

    user_folder = f"user_{USER_NAME}"
    raw_data_folder = os.path.join(user_folder, "raw_data")
    os.makedirs(raw_data_folder, exist_ok=True)

    standard_rows = []
    for i, sample in enumerate(SESSION_SAMPLES):
        start = sample['start']
        mid = sample['mid']
        end = sample['end']

        row = [
            i + 1,
            pose_label,
            0, 0, 0, 0, 0,
            sample['finger_thumb'],
            sample['finger_index'],
            sample['finger_middle'],
            sample['finger_ring'],
            sample['finger_pinky'],
            float(start[0]), float(start[1]),
            float(mid[0]), float(mid[1]),
            float(end[0]), float(end[1]),
            sample['main_axis_x'],
            sample['main_axis_y'],
            sample['motion_x'],
            sample['motion_y']
        ]
        standard_rows.append(row)

    columns = [
        'instance_id', 'pose_label',
        'left_finger_state_0', 'left_finger_state_1', 'left_finger_state_2', 'left_finger_state_3', 'left_finger_state_4',
        'right_finger_state_0', 'right_finger_state_1', 'right_finger_state_2', 'right_finger_state_3', 'right_finger_state_4',
        'motion_x_start', 'motion_y_start', 'motion_x_mid', 'motion_y_mid', 'motion_x_end', 'motion_y_end',
        'main_axis_x', 'main_axis_y', 'delta_x', 'delta_y'
    ]

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_csv = os.path.join(raw_data_folder, f"gesture_data_custom_{USER_NAME}_{pose_label}_{timestamp}.csv")

    individual_df = pd.DataFrame(standard_rows, columns=columns)
    individual_df.to_csv(user_csv, index=False)

    print(f"\n[SAVE] Saved {len(SESSION_SAMPLES)} samples (standard format) to: {user_csv}")

    update_master_csv_in_user_folder(pose_label, standard_rows, columns)

    return user_csv


def update_master_csv_in_user_folder(pose_label, standard_rows, columns):
    """Update master CSV file in user folder with standard format"""
    user_folder = f"user_{USER_NAME}"
    master_csv = os.path.join(user_folder, f"gesture_data_custom_{USER_NAME}.csv")

    start_id = 1
    if os.path.exists(master_csv):
        try:
            existing_df = pd.read_csv(master_csv)
            if not existing_df.empty and 'instance_id' in existing_df.columns:
                start_id = existing_df['instance_id'].max() + 1
        except Exception:
            start_id = 1

    for i, row in enumerate(standard_rows):
        row[0] = start_id + i

    new_df = pd.DataFrame(standard_rows, columns=columns)

    if os.path.exists(master_csv):
        new_df.to_csv(master_csv, mode='a', header=False, index=False)
        print(f"Appended {len(standard_rows)} samples to master file: {master_csv}")
    else:
        new_df.to_csv(master_csv, index=False)
        print(f"Created new master file: {master_csv} with {len(standard_rows)} samples")

# =========================
#  MAIN LOOP FOR WINFORMS
# =========================
def main():
    global USER_NAME, SESSION_SAMPLES, QUALITY_SAMPLES

    # 1. Nhận user & pose từ WinForms
    USER_NAME, pose_label = receive_user_and_pose()
    if not USER_NAME or not pose_label:
        print("No user or pose label provided!")
        sys.exit(1)

    # 2. Load dữ liệu user & ref cho conflict detection
    load_user_gesture_data(USER_NAME)
    load_reference_data()

    # 3. Kết nối WinForms
    cam_conn = start_camera_server()
    status_conn = start_status_server()

    SESSION_SAMPLES = []
    QUALITY_SAMPLES = []

    cap = cv2.VideoCapture(0)
    state = 'WAIT'
    buffer = collections.deque(maxlen=BUFFER_SIZE)
    saved_count = 0
    current_left_states = None
    current_right_states = None
    conflict_detected = False

    try:
        while cap.isOpened():
            if not is_connection_alive(cam_conn) or not is_connection_alive(status_conn):
                print("Connection lost. WinForms closed. Exiting...")
                break

            ok, frame = cap.read()
            if not ok:
                print('[ERROR] Camera frame error!')
                break

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

            # Gửi frame cho WinForms hiển thị
            send_frame_to_winforms(cam_conn, frame)

            left_is_fist = is_fist(left_landmarks)

            if state == 'WAIT':
                if left_is_fist and left_conf > MIN_CONFIDENCE and right_conf > MIN_CONFIDENCE and right_landmarks:
                    current_left_states = get_finger_states(left_landmarks, 'Left')
                    current_right_states = get_finger_states(right_landmarks, 'Right')
                    buffer.clear()
                    state = 'RECORD'
                    send_status_to_winforms(status_conn, "START", pose_label, saved_count, conflict_detected, "START")
                elif left_is_fist and right_landmarks is None:
                    send_status_to_winforms(status_conn, "ERROR", pose_label, saved_count, False, "ERROR_RIGHT_HAND")

            elif state == 'RECORD':
                if right_landmarks and right_conf > MIN_CONFIDENCE:
                    wrist = right_landmarks.landmark[0]
                    buffer.append(np.array([wrist.x, wrist.y], dtype=float))
                if not left_is_fist:
                    state = 'PROCESS'

            elif state == 'PROCESS':
                if len(buffer) < MIN_FRAMES or current_right_states is None:
                    send_status_to_winforms(
                        status_conn, "ERROR", pose_label, saved_count,
                        False, "ERROR_INSUFFICIENT_DATA", len(buffer)
                    )
                else:
                    smoothed = smooth_points(list(buffer))
                    features = compute_motion_features(smoothed)
                    if features is None:
                        send_status_to_winforms(
                            status_conn, "ERROR", pose_label, saved_count,
                            False, "ERROR_MOTION_FEATURES"
                        )
                    else:
                        sample_data = {
                            'left_states': current_left_states if current_left_states else [0, 0, 0, 0, 0],
                            'right_states': current_right_states,
                            'features': features
                        }

                        has_conflict, conflict_msg, _ = check_realtime_conflict(
                            sample_data, pose_label, username=USER_NAME
                        )

                        if has_conflict:
                            print(f"CONFLICT: {conflict_msg}")
                            send_status_to_winforms(
                                status_conn, "CONFLICT", pose_label, saved_count,
                                True, conflict_msg
                            )
                            conflict_detected = True
                        else:
                            # Build user_row & lưu tạm để validate
                            user_row = {
                                'finger_thumb': current_right_states[0],
                                'finger_index': current_right_states[1],
                                'finger_middle': current_right_states[2],
                                'finger_ring': current_right_states[3],
                                'finger_pinky': current_right_states[4],
                                'start': features['start'],
                                'mid': features['mid'],
                                'end': features['end'],
                                'main_axis_x': features['main_axis_x'],
                                'main_axis_y': features['main_axis_y'],
                                'motion_x': features['delta_x'],
                                'motion_y': features['delta_y']
                            }

                            QUALITY_SAMPLES.append(user_row.copy())
                            SESSION_SAMPLES.append(user_row)
                            saved_count += 1
                            conflict_detected = False

                            send_status_to_winforms(
                                status_conn, "COLLECTED", pose_label, saved_count,
                                False, "SAMPLE_COLLECTED"
                            )

                            # Đủ 5 sample -> quality validation
                            if saved_count >= REQUIRED_SAMPLES:
                                is_valid, consistent_samples, message = validate_sample_quality(QUALITY_SAMPLES)

                                if is_valid:
                                    SESSION_SAMPLES = consistent_samples
                                    send_status_to_winforms(
                                        status_conn, "FINISH", pose_label, saved_count,
                                        False, "FINISH"
                                    )
                                    break
                                else:
                                    print(f"[QUALITY FAILED] {message}")
                                    send_status_to_winforms(
                                        status_conn, "ERROR", pose_label, saved_count,
                                        True, "QUALITY_FAILED", message
                                    )
                                    break

                buffer.clear()
                current_left_states = None
                current_right_states = None
                state = 'WAIT'

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nEXIT. Samples collected: {saved_count}")

        if len(SESSION_SAMPLES) > 0:
            user_csv_path = save_session_to_user_folder(pose_label)
            if user_csv_path:
                print(f"[OK] Saved to: {user_csv_path}")
                try:
                    send_status_to_winforms(
                        status_conn, "FINISH", pose_label, saved_count,
                        False, "SAVED_FILE", user_csv_path
                    )
                except Exception as e:
                    print("[WARN] Cannot send final status:", e)

        try:
            cam_conn.close()
        except:
            pass
        try:
            status_conn.close()
        except:
            pass


if __name__ == '__main__':
    main()