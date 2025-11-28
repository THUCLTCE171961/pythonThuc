# #!/usr/bin/env python3
# """
# Simple script to prepare user training data
# Usage: python prepare_user_data.py user_folder
# Example: python prepare_user_data.py user_Bi
# """

# import pandas as pd
# import numpy as np
# import sys
# from pathlib import Path

# def prepare_user_training_data(user_folder):
#     """Prepare complete training data for user"""
    
#     user_path = Path(user_folder)
#     if not user_path.exists():
#         print(f"❌ User folder not found: {user_folder}")
#         return False
    
#     # 1. Find user's custom data file
#     custom_files = list(user_path.glob("gesture_data_custom_*.csv"))
#     if not custom_files:
#         print(f"❌ No custom data file found in {user_folder}")
#         print("   Expected: gesture_data_custom_*.csv")
#         return False
    
#     user_data_file = custom_files[0]
#     print(f"📂 Found user data: {user_data_file}")
    
#     # 2. Load base reference data
#     base_file = Path("training_results/gesture_data_compact.csv")
#     if not base_file.exists():
#         print(f"❌ Base reference file not found: {base_file}")
#         return False
    
#     print(f"📂 Loading base data: {base_file}")
#     base_df = pd.read_csv(base_file)
#     user_df = pd.read_csv(user_data_file)
    
#     user_gestures = user_df['pose_label'].unique()
#     print(f"✅ User has {len(user_gestures)} custom gestures: {list(user_gestures)}")
    
#     # 3. Create compact dataset (10 samples)
#     print(f"\n🔄 Creating compact dataset...")
#     compact_samples = []
    
#     for _, base_row in base_df.iterrows():
#         gesture = base_row['pose_label']
        
#         if gesture in user_gestures:
#             # Use user's custom data
#             user_samples = user_df[user_df['pose_label'] == gesture]
#             selected = user_samples.iloc[0].copy()  # Use first sample
#             selected['instance_id'] = len(compact_samples) + 1
#             print(f"   ✅ {gesture}: Using CUSTOM data")
#         else:
#             # Use base data
#             selected = base_row.copy()
#             selected['instance_id'] = len(compact_samples) + 1
#             print(f"   📋 {gesture}: Using BASE data")
        
#         compact_samples.append(selected)
    
#     compact_df = pd.DataFrame(compact_samples)
    
#     # 4. Create training_results directory
#     training_dir = user_path / "training_results"
#     training_dir.mkdir(exist_ok=True)
    
#     # Save compact dataset
#     compact_file = training_dir / "gesture_data_compact.csv"
#     compact_df.to_csv(compact_file, index=False)
#     print(f"✅ Compact dataset saved: {compact_file}")
    
#     # 5. Generate balanced dataset (1000 samples)
#     print(f"\n🔄 Generating balanced dataset (1000 samples)...")
#     balanced_samples = []
    
#     np.random.seed(42)  # For reproducibility
    
#     for gesture in compact_df['pose_label'].unique():
#         base_row = compact_df[compact_df['pose_label'] == gesture].iloc[0]
        
#         for i in range(100):  # 100 samples per gesture
#             new_row = base_row.copy()
#             new_row['instance_id'] = len(balanced_samples) + 1
            
#             # Add small noise to motion features
#             motion_cols = ['motion_x_start', 'motion_y_start', 'motion_x_mid', 
#                           'motion_y_mid', 'motion_x_end', 'motion_y_end', 
#                           'main_axis_x', 'main_axis_y', 'delta_x', 'delta_y']
            
#             for col in motion_cols:
#                 if col in new_row:
#                     new_row[col] += np.random.normal(0, 0.01)
            
#             balanced_samples.append(new_row)
    
#     balanced_df = pd.DataFrame(balanced_samples)
    
#     # Save balanced dataset
#     balanced_file = user_path / "gesture_data_1000_balanced.csv"
#     balanced_df.to_csv(balanced_file, index=False)
#     print(f"✅ Balanced dataset saved: {balanced_file}")
    
#     # 6. Auto-train user models
#     print(f"\n🚀 Auto-training user models...")
    
#     # Import training functions
#     import subprocess
#     import os
    
#     try:
#         # Run training command
#         cmd = f"python train_user_models.py --dataset {balanced_file}"
#         print(f"   Running: {cmd}")
        
#         result = subprocess.run(cmd.split(), 
#                               capture_output=True, 
#                               text=True, 
#                               cwd=os.getcwd())
        
#         if result.returncode == 0:
#             print(f"✅ Training completed successfully!")
#             print(f"📊 Model saved to: {user_path}/models/")
#         else:
#             print(f"❌ Training failed:")
#             print(result.stderr)
    
#     except Exception as e:
#         print(f"❌ Training error: {e}")
#         print(f"🔧 Manual train: python train_user_models.py --dataset {balanced_file}")
    
#     # Summary
#     print(f"\n🎉 User setup completed!")
#     print(f"   📊 Templates: {len(compact_df)} gestures → {compact_file}")
#     print(f"   📊 Training: {len(balanced_df)} samples → {balanced_file}")
#     print(f"   🤖 Models: {user_path}/models/")
    
#     return True

# def main():
#     if len(sys.argv) != 2:
#         print("Usage: python prepare_user_data.py <user_folder>")
#         print("Example: python prepare_user_data.py user_Bi")
#         sys.exit(1)
    
#     user_folder = sys.argv[1]
#     success = prepare_user_training_data(user_folder)
    
#     if not success:
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Chuẩn bị dữ liệu huấn luyện tùy biến cho từng admin/user.

Luồng mới:
    python prepare_user_data.py --user-id 123 --custom-csv path/to.csv

Luồng cũ (tương thích) - chỉ tạo dữ liệu, không train:
    python prepare_user_data.py user_Khang

Để train luôn sau khi tạo dữ liệu:
    python prepare_user_data.py user_Khang --train
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_COMPACT = SCRIPT_DIR / "training_results" / "gesture_data_compact.csv"
DEFAULT_ORIGINAL_DATA = SCRIPT_DIR / "gesture_data_09_10_2025.csv"

CUSTOM_SAMPLES = 100
CUSTOM_ERROR_RATIO = 0.25
DEFAULT_SYNTH_SAMPLES = 80
RANDOM_SEED = 42

# Constants for custom data generation
TOTAL_CUSTOM_SAMPLES = 225  # 200-275, chọn 225
ACCURATE_RATIO = 0.75  # 75% chính xác, 25% có nhiễu


def analyze_gesture_pattern(df: pd.DataFrame, gesture: str) -> dict:
    """Phân tích pattern của gesture để tạo nhiễu thực tế."""
    gesture_df = df[df["pose_label"] == gesture]
    if gesture_df.empty:
        return {}
    
    pattern = {}
    
    # Finger states: mode (most common)
    finger_cols = [f"right_finger_state_{i}" for i in range(5)]
    pattern["finger_mode"] = []
    for col in finger_cols:
        if col in gesture_df.columns:
            mode_val = gesture_df[col].mode()
            pattern["finger_mode"].append(mode_val.iloc[0] if not mode_val.empty else 0)
        else:
            pattern["finger_mode"].append(0)
    
    # Motion vectors: mean and std
    motion_cols = ["delta_x", "delta_y"]
    for col in motion_cols:
        if col in gesture_df.columns:
            pattern[f"{col}_mean"] = gesture_df[col].mean()
            pattern[f"{col}_std"] = gesture_df[col].std()
    
    # Direction if available
    if "direction" in gesture_df.columns:
        pattern["direction_mean"] = gesture_df["direction"].mean()
        pattern["direction_std"] = gesture_df["direction"].std()
    
    return pattern


def add_gesture_specific_noise(df: pd.DataFrame, pattern: dict) -> pd.DataFrame:
    """Thêm nhiễu thực tế dựa trên pattern của gesture."""
    noisy_df = df.copy()
    
    # Finger states: flip 1-2 bits so với mode (như lỗi thực tế)
    if "finger_mode" in pattern:
        finger_cols = [f"right_finger_state_{i}" for i in range(5)]
        for idx, row in noisy_df.iterrows():
            # Flip 1-2 fingers randomly
            flip_count = np.random.choice([1, 2])
            flip_indices = np.random.choice(5, flip_count, replace=False)
            for i in flip_indices:
                col = finger_cols[i]
                if col in noisy_df.columns:
                    noisy_df.at[idx, col] = 1 - row[col]  # Flip
    
    # Motion vectors: thêm noise với std từ pattern
    for col in ["delta_x", "delta_y"]:
        if f"{col}_std" in pattern and col in noisy_df.columns:
            std_val = pattern[f"{col}_std"]
            if std_val > 0:
                noise = np.random.normal(0, std_val * 0.5, len(noisy_df))  # Nhiễu nhỏ hơn std
                noisy_df[col] += noise
    
    # Direction nếu có
    if "direction_std" in pattern and "direction" in noisy_df.columns:
        std_val = pattern["direction_std"]
        if std_val > 0:
            noise = np.random.normal(0, std_val * 0.1, len(noisy_df))
            noisy_df["direction"] += noise
    
    return noisy_df


def resolve_user_path(args: argparse.Namespace) -> Path:
    """Xác định thư mục user sẽ chứa kết quả."""
    if args.user_dir:
        return Path(args.user_dir).resolve()
    if args.user_folder:
        return Path(args.user_folder).resolve()
    if args.user_id:
        base = Path(args.output_root).resolve() if args.output_root else SCRIPT_DIR
        return (base / f"user_{args.user_id}").resolve()
    raise ValueError("Cần cung cấp --user-dir, --user-id hoặc đối số user_folder (legacy).")


def merge_user_csvs(user_path: Path) -> Path | None:
    """Gộp tất cả file CSV từ raw_data/ thành một file master, với logic đặc biệt cho user data"""
    raw_data_path = user_path / "raw_data"

    if not raw_data_path.exists():
        return None

    all_dfs = []
    instance_id = 1

    # Duyệt qua tất cả thư mục con trong raw_data
    for subdir in raw_data_path.iterdir():
        if subdir.is_dir():
            # Tìm file CSV trong thư mục con
            csv_files = list(subdir.glob("gesture_data_custom_*.csv"))
            for csv_file in csv_files:
                print(f"[MERGE] Đọc file: {csv_file}")
                df = pd.read_csv(csv_file)
                # Cập nhật instance_id
                df['instance_id'] = range(instance_id, instance_id + len(df))
                instance_id += len(df)
                all_dfs.append(df)

    if not all_dfs:
        return None

    # Gộp tất cả DataFrames
    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Lưu file master
    master_csv = user_path / f"gesture_data_custom_{user_path.name}.csv"
    merged_df.to_csv(master_csv, index=False)
    print(f"[MERGE] Đã tạo file master: {master_csv} với {len(merged_df)} mẫu")

    return master_csv


def create_enhanced_user_dataset(user_path: Path, custom_csv: Path, reference_csv: Path) -> Path:
    """Tạo dataset enhanced: loại bỏ custom gestures từ reference, tạo custom data với nhiễu thực tế."""
    # Load user data và reference data
    user_df = pd.read_csv(custom_csv)
    ref_df = pd.read_csv(reference_csv)

    print(f"[ENHANCE] User data: {len(user_df)} samples")
    print(f"[ENHANCE] Reference data: {len(ref_df)} samples")

    # Lấy user gestures
    user_gestures = set(user_df["pose_label"].unique())
    print(f"[ENHANCE] Custom gestures: {sorted(user_gestures)}")

    enhanced_samples = []

    # Xử lý từng gesture
    for gesture in sorted(ref_df["pose_label"].unique()):
        if gesture in user_gestures:
            # User có custom data cho gesture này
            user_gesture_data = user_df[user_df["pose_label"] == gesture].copy()
            original_count = len(user_gesture_data)

            # Phân tích pattern từ user data
            pattern = analyze_gesture_pattern(user_df, gesture)
            print(f"[ENHANCE] {gesture} pattern: finger_mode={pattern.get('finger_mode', [])}")

            # Tạo custom samples: 75% chính xác, 25% có nhiễu
            accurate_count = int(TOTAL_CUSTOM_SAMPLES * ACCURATE_RATIO)
            noise_count = TOTAL_CUSTOM_SAMPLES - accurate_count

            # Samples chính xác: duplicate user data
            duplicated_accurate = []
            accurate_per_template = accurate_count // original_count
            for i in range(accurate_per_template):
                temp_df = user_gesture_data.copy()
                duplicated_accurate.append(temp_df)
            # Thêm phần dư nếu có
            remaining = accurate_count % original_count
            if remaining > 0:
                temp_df = user_gesture_data.head(remaining).copy()
                duplicated_accurate.append(temp_df)
            accurate_df = pd.concat(duplicated_accurate, ignore_index=True)

            # Samples có nhiễu: duplicate với nhiễu gesture-specific
            duplicated_noise = []
            noise_per_template = noise_count // original_count
            for i in range(noise_per_template):
                temp_df = user_gesture_data.copy()
                # Thêm nhiễu thực tế dựa trên pattern
                temp_df = add_gesture_specific_noise(temp_df, pattern)
                duplicated_noise.append(temp_df)
            # Thêm phần dư
            remaining_noise = noise_count % original_count
            if remaining_noise > 0:
                temp_df = user_gesture_data.head(remaining_noise).copy()
                temp_df = add_gesture_specific_noise(temp_df, pattern)
                duplicated_noise.append(temp_df)
            noise_df = pd.concat(duplicated_noise, ignore_index=True)

            # Combine accurate + noise
            enhanced_gesture = pd.concat([accurate_df, noise_df], ignore_index=True)
            enhanced_samples.append(enhanced_gesture)
            print(f"[ENHANCE] {gesture}: {original_count} -> {len(enhanced_gesture)} samples ({accurate_count} accurate, {noise_count} with noise)")

        else:
            # Dùng reference data, loại bỏ custom gestures (đã được xử lý ở trên)
            ref_gesture_data = ref_df[ref_df["pose_label"] == gesture].copy()
            enhanced_samples.append(ref_gesture_data)
            print(f"[ENHANCE] {gesture}: {len(ref_gesture_data)} samples (reference)")

    # Combine all
    final_df = pd.concat(enhanced_samples, ignore_index=True)
    
    # Reset instance_id theo thứ tự từ 0
    final_df['instance_id'] = range(len(final_df))

    # Save enhanced dataset
    enhanced_csv = user_path / "gesture_data_custom_full.csv"
    final_df.to_csv(enhanced_csv, index=False)

    print(f"[ENHANCE] Created enhanced dataset: {enhanced_csv}")
    print(f"[ENHANCE] Total samples: {len(final_df)}")
    print(f"[ENHANCE] Gestures: {sorted(final_df['pose_label'].unique())}")

    return enhanced_csv
def ensure_custom_csv(user_path: Path, custom_csv: str | None) -> Path:
    """Đảm bảo có file dữ liệu custom và copy vào folder user nếu cần."""
    if custom_csv:
        src = Path(custom_csv).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Không tìm thấy file custom CSV: {src}")
        user_path.mkdir(parents=True, exist_ok=True)
        dest = user_path / src.name
        if dest != src:
            shutil.copy2(src, dest)
            print(f"[INFO] Đã copy file custom vào {dest}")
        else:
            print(f"[INFO] Sử dụng file custom có sẵn: {dest}")
        return dest

    candidates = sorted(user_path.glob("gesture_data_custom_*.csv"))
    if not candidates:
        # Không tìm thấy file custom trực tiếp, thử merge từ raw_data
        print(f"[INFO] Không tìm thấy file custom trực tiếp, thử merge từ raw_data...")
        merged_csv = merge_user_csvs(user_path)
        if merged_csv:
            return merged_csv
        else:
            raise FileNotFoundError(
                f"Không tìm thấy file custom trong {user_path} hoặc raw_data/. Cần file theo mẫu gesture_data_custom_*.csv"
            )
    print(f"[INFO] Phát hiện file custom: {candidates[0]}")
    return candidates[0]


def load_dataframe(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} không tồn tại: {path}")
    print(f"[LOAD] {label}: {path}")
    return pd.read_csv(path)


def create_custom_dataset(base_df: pd.DataFrame, user_df: pd.DataFrame, original_df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Tạo dataset tùy chỉnh: copy tất cả samples từ original cho gestures mặc định, override custom gestures với 100 mẫu."""
    user_gestures = set(user_df["pose_label"].unique())
    samples: list[pd.Series] = []

    print("\n[STEP] Tạo custom dataset...")
    
    # Copy tất cả samples từ original dataset cho mỗi gesture
    for gesture in original_df["pose_label"].unique():
        if gesture in user_gestures:
            # Override với custom gesture: tạo samples từ TẤT CẢ templates custom có sẵn
            # Mỗi template tạo ra nhiều mẫu với noise
            gesture_templates = user_df[user_df["pose_label"] == gesture]
            total_templates = len(gesture_templates)
            
            # Tăng số samples cho mỗi template (từ 100 xuống ~50-60 mỗi template)
            samples_per_template = max(50, CUSTOM_SAMPLES // total_templates)
            total_samples = samples_per_template * total_templates
            
            print(f"   [CUSTOM] {gesture}: {total_templates} templates -> {total_samples} mẫu tổng cộng")
            print(f"      Mỗi template tạo {samples_per_template} mẫu (75% chính xác, 25% có noise)")
            
            np.random.seed(RANDOM_SEED)
            sample_idx = 0
            
            for template_idx, (_, template) in enumerate(gesture_templates.iterrows()):
                error_count = int(samples_per_template * 0.3)  # 30% có noise
                
                for local_idx in range(samples_per_template):
                    has_error = local_idx < error_count
                    new_row = add_noise(template.copy(), has_error)
                    new_row["instance_id"] = len(samples) + 1
                    new_row["pose_label"] = gesture
                    samples.append(new_row)
                    sample_idx += 1
        else:
            # Copy tất cả samples của gesture mặc định từ original
            gesture_samples = original_df[original_df["pose_label"] == gesture]
            print(f"   [DEFAULT] {gesture}: copy {len(gesture_samples)} mẫu từ original dataset")
            
            for _, sample in gesture_samples.iterrows():
                sample_copy = sample.copy()
                sample_copy["instance_id"] = len(samples) + 1
                samples.append(sample_copy)

    custom_df = pd.DataFrame(samples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    custom_df.to_csv(out_path, index=False)
    print(f"[SAVED] Custom dataset -> {out_path} ({len(custom_df)} mẫu)")
    return custom_df


def add_noise(row: pd.Series, error_mode: bool) -> pd.Series:
    """Thêm nhiễu để mô phỏng lỗi khi người dùng thực hiện gesture."""
    noisy = row.copy()
    if error_mode:
        if np.random.random() < 0.5:
            finger_cols = [f"right_finger_state_{i}" for i in range(5)]
            for _ in range(np.random.choice([1, 2])):
                col = np.random.choice(finger_cols)
                noisy[col] = 1 - noisy[col]

        for col in ("delta_x", "delta_y"):
            if col in noisy:
                noisy[col] += np.random.normal(0, 0.05)

        if np.random.random() < 0.2:
            if "main_axis_x" in noisy:
                noisy["main_axis_x"] = 1 - noisy["main_axis_x"]
            if "main_axis_y" in noisy:
                noisy["main_axis_y"] = 1 - noisy["main_axis_y"]
    else:
        motion_cols = [
            "motion_x_start",
            "motion_y_start",
            "motion_x_mid",
            "motion_y_mid",
            "motion_x_end",
            "motion_y_end",
            "delta_x",
            "delta_y",
        ]
        for col in motion_cols:
            if col in noisy:
                noisy[col] += np.random.normal(0, 0.008)
    return noisy


def original_samples_for_gesture(original_df: pd.DataFrame | None, gesture: str) -> Iterable[pd.Series]:
    if original_df is None:
        return []
    subset = original_df[original_df["pose_label"] == gesture]
    return subset.itertuples(index=False, name=None) if not subset.empty else []


def create_balanced_dataset(
    compact_df: pd.DataFrame,
    user_gestures: set[str],
    original_df: pd.DataFrame | None,
    out_path: Path,
) -> pd.DataFrame:
    """DEPRECATED: Không dùng nữa, thay bằng create_custom_dataset"""
    print("[WARN] create_balanced_dataset is deprecated, using create_custom_dataset instead")
    return create_custom_dataset(compact_df, pd.DataFrame(), out_path)


def run_training(custom_file: Path, user_path: Path, skip_training: bool) -> None:
    if skip_training:
        print("\n[TRAINING] Bỏ qua bước train (do dùng --skip-training).")
        return

    # Copy train_motion_svm_all_models.py vào user folder và chỉnh đường dẫn
    user_train_script = user_path / "train_motion_svm_all_models.py"
    original_train_script = SCRIPT_DIR / "train_motion_svm_all_models.py"
    
    if not original_train_script.exists():
        print(f"[ERROR] Không tìm thấy script train gốc: {original_train_script}")
        return
    
    # Copy script
    shutil.copy2(original_train_script, user_train_script)
    print(f"[COPY] Đã copy train script vào: {user_train_script}")
    
    # Chỉnh sửa script để lưu models và training_results vào user folder
    with open(user_train_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Thay đổi BASE_DIR, RESULTS_DIR, MODELS_DIR
    user_dir_str = str(user_path)
    content = content.replace(
        'BASE_DIR = os.path.dirname(os.path.abspath(__file__))',
        f'BASE_DIR = r"{user_dir_str}"'
    )
    content = content.replace(
        'RESULTS_DIR = Path(BASE_DIR) / "training_results"',
        f'RESULTS_DIR = Path(r"{user_dir_str}") / "training_results"'
    )
    content = content.replace(
        'MODELS_DIR = Path(BASE_DIR) / "models"',
        f'MODELS_DIR = Path(r"{user_dir_str}") / "models"'
    )
    
    # Thay đổi DEFAULT_DATASET để dùng custom_file
    custom_file_str = str(custom_file)
    content = content.replace(
        'DEFAULT_DATASET = os.path.join(BASE_DIR, "gesture_motion_dataset_realistic.csv")',
        f'DEFAULT_DATASET = r"{custom_file_str}"'
    )
    
    with open(user_train_script, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[MODIFY] Đã chỉnh sửa script để lưu vào user folder")
    
    # Chạy script đã chỉnh sửa
    cmd = [sys.executable, str(user_train_script)]
    print("\n[TRAINING] Chạy:", " ".join(cmd))
    print("=" * 60)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(user_path),  # Chạy trong user folder
        bufsize=1,
    )

    logs: list[str] = []
    while True:
        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if line:
            clean = line.rstrip()
            print(clean)
            logs.append(clean)

    code = process.poll()
    print("=" * 60)
    if code != 0:
        print(f"[ERROR] Train thất bại, exit code {code}")
        print(f"[HINT] Tự chạy lại: python {user_train_script}")
        return

    summary = [l for l in logs if "F1-score" in l or "accuracy" in l or "TRAINING COMPLETE" in l]
    if summary:
        print("\n[SUMMARY]")
        for item in summary:
            print("   " + item)
    print("[SUCCESS] Train hoàn tất.")


def prepare_user_training(args: argparse.Namespace) -> bool:
    try:
        user_path = resolve_user_path(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return False

    user_path.mkdir(parents=True, exist_ok=True)
    try:
        custom_csv = ensure_custom_csv(user_path, args.custom_csv)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return False

    base_path = Path(args.base_compact).resolve() if args.base_compact else DEFAULT_BASE_COMPACT
    original_path = Path(args.original_data).resolve() if args.original_data else DEFAULT_ORIGINAL_DATA

    try:
        base_df = load_dataframe(base_path, "Base compact dataset")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return False

    try:
        user_df = load_dataframe(custom_csv, "Custom dataset")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return False

    if original_path.exists():
        original_df = pd.read_csv(original_path)
        print(f"[INFO] Original dataset: {len(original_df)} mẫu từ {original_path}")
    else:
        print(f"[WARN] Không tìm thấy original dataset ({original_path}). Sẽ sinh dữ liệu bằng noise.")
        original_df = None

    compact_file = user_path / "training_results" / "gesture_data_compact.csv"
    custom_file = user_path / "gesture_data_custom_full.csv"

    # Tạo enhanced dataset: duplicate user data + merge với reference
    if original_path.exists():
        enhanced_file = create_enhanced_user_dataset(user_path, custom_csv, original_path)
        if enhanced_file:
            custom_df = pd.read_csv(enhanced_file)
        else:
            print("[ERROR] Không thể tạo enhanced dataset")
            return False
    else:
        print("[ERROR] Cần file reference data để tạo enhanced dataset")
        return False

    # Mặc định LUÔN skip training, chỉ prepare dataset
    # Chỉ train khi user chỉ định --train
    if args.train:
        run_training(custom_file, user_path, False)  # Chạy training nếu user muốn
    else:
        print("\n[SKIP] Bỏ qua training. Chạy riêng sau:")
        print(f"   cd {user_path}")
        print(f"   python train_motion_svm_all_models.py")

    print("\n[DONE]")
    print(f"   Custom  : {custom_file} ({len(custom_df)} dòng)")
    if args.train:
        print(f"   Models  : {user_path / 'models'}")
        print(f"   Results : {user_path / 'training_results'}")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu và train model cho user gesture.")
    parser.add_argument(
        "user_folder",
        nargs="?",
        help="(Legacy) thư mục user sẵn có (ví dụ user_Bi).",
    )
    parser.add_argument("--user-id", help="ID user/admin (sẽ tạo folder user_<id>).")
    parser.add_argument("--user-dir", help="Đường dẫn tuyệt đối tới thư mục user.")
    parser.add_argument("--output-root", help="Thư mục cha để tạo user_<id> nếu dùng --user-id.")
    parser.add_argument("--custom-csv", help="Đường dẫn file CSV custom vừa thu thập.")
    parser.add_argument("--base-compact", help="Đường dẫn file compact gốc.")
    parser.add_argument("--original-data", help="Đường dẫn dataset mặc định đầy đủ.")
    parser.add_argument("--train", action="store_true", help="Chạy training sau khi tạo dữ liệu.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    success = prepare_user_training(args)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
