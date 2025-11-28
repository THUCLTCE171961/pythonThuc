import os
import pickle
from pathlib import Path

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# === Config ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(BASE_DIR, "gesture_motion_dataset_realistic.csv")
RESULTS_DIR = Path(BASE_DIR)

LEFT_COLS = [f"left_finger_state_{i}" for i in range(5)]
RIGHT_COLS = [f"right_finger_state_{i}" for i in range(5)]
MOTION_COLS = ["main_axis_x", "main_axis_y", "delta_x", "delta_y"]

DELTA_WEIGHT = 15.0  # Increased from 5.0 to emphasize motion direction
MIN_DELTA_MAG = 0.05

COARSE_C_VALUES = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3]
COARSE_GAMMA_VALUES = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, "scale", "auto"]
FINE_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]

MODEL_PKL = os.path.join(BASE_DIR, "motion_svm_model.pkl")
SCALER_PKL = os.path.join(BASE_DIR, "motion_scaler.pkl")


TEST_FRACTION = 0.25
PREDICT_MIN_PROB = 0.6


# === Data utilities ===
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing dataset: {path}")
    df = pd.read_csv(path)
    missing_cols = [col for col in LEFT_COLS + RIGHT_COLS + MOTION_COLS + ["pose_label", "base_instance_id", "instance_id"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    return df


def prepare_features(df: pd.DataFrame):
    df = df.copy()

    # enforce numeric types
    for col in LEFT_COLS + RIGHT_COLS:
        df[col] = df[col].fillna(0).astype(int)

    delta_x_series = df["delta_x"].astype(float)
    delta_y_series = df["delta_y"].astype(float)
    axis_x = (delta_x_series.abs() >= delta_y_series.abs()).astype(int)
    axis_y = 1 - axis_x
    df["main_axis_x"] = axis_x.astype(float)
    df["main_axis_y"] = axis_y.astype(float)
    df.loc[axis_x == 1, "delta_x"] = delta_x_series[axis_x == 1]
    df.loc[axis_x == 1, "delta_y"] = 0.0
    df.loc[axis_x == 0, "delta_y"] = delta_y_series[axis_x == 0]
    df.loc[axis_x == 0, "delta_x"] = 0.0

    df["delta_mag"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2)
    before = len(df)
    df = df[df["delta_mag"] >= MIN_DELTA_MAG].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[INFO] Dropped {dropped} samples with delta_mag < {MIN_DELTA_MAG}")
    df = df.drop(columns=["delta_mag"])

    df.loc[:, "delta_x"] = df["delta_x"] * DELTA_WEIGHT
    df.loc[:, "delta_y"] = df["delta_y"] * DELTA_WEIGHT
    
    # Add explicit direction features to help distinguish similar finger states
    df["motion_left"] = (df["delta_x"] < 0).astype(float) * DELTA_WEIGHT
    df["motion_right"] = (df["delta_x"] > 0).astype(float) * DELTA_WEIGHT
    df["motion_up"] = (df["delta_y"] < 0).astype(float) * DELTA_WEIGHT  
    df["motion_down"] = (df["delta_y"] > 0).astype(float) * DELTA_WEIGHT

    finger_feats = df[LEFT_COLS + RIGHT_COLS].values.astype(float)
    motion_feats = df[MOTION_COLS + ["motion_left", "motion_right", "motion_up", "motion_down"]].values.astype(float)

    scaler = StandardScaler()
    motion_scaled = scaler.fit_transform(motion_feats)

    X = np.hstack([finger_feats, motion_scaled])
    labels = df["pose_label"].values
    groups = df["base_instance_id"].astype(int).values

    return X, labels, scaler, groups


def stratified_group_split(labels: np.ndarray, groups: np.ndarray, test_fraction: float, random_state: int = 42):
    data = pd.DataFrame({'group': groups, 'label': labels})
    group_labels = data.drop_duplicates('group')

    group_counts = group_labels.groupby('label')['group'].nunique()
    insufficient = group_counts[group_counts < 2]
    if not insufficient.empty:
        details = ', '.join(f"{label}: {count}" for label, count in insufficient.items())
        raise ValueError(f"Can them mau cho cac pose -> {details}")

    rng = np.random.default_rng(random_state)
    test_groups = []

    for pose, pose_groups in group_labels.groupby('label')['group']:
        pose_group_ids = pose_groups.to_numpy()
        total_groups = len(pose_group_ids)
        n_test = max(1, int(round(total_groups * test_fraction)))
        if n_test >= total_groups:
            n_test = total_groups - 1
        selected = rng.choice(pose_group_ids, size=n_test, replace=False)
        test_groups.extend(selected.tolist())

    test_mask = np.isin(groups, test_groups)
    test_idx = np.flatnonzero(test_mask)
    train_idx = np.flatnonzero(~test_mask)
    return train_idx, test_idx


# === Grid search utilities ===
def build_fine_values(best_value, multipliers):
    if isinstance(best_value, str):
        return [best_value]
    fine_set = set()
    for m in multipliers:
        candidate = best_value * m
        if candidate > 0:
            fine_set.add(candidate)
    return sorted(fine_set)


def run_grid_search(description: str,
                    estimator: SVC,
                    X: np.ndarray,
                    y: np.ndarray,
                    groups: np.ndarray,
                    kernels,
                    Cs,
                    gammas,
                    output_name: str):
    print(f"\n=== {description} ===")
    param_grid = {
        "kernel": kernels,
        "C": Cs,
        "gamma": gammas,
    }
    cv = GroupKFold(n_splits=10)
    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X, y, groups=groups)

    results = pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False)
    display_cols = ["mean_test_score", "std_test_score", "param_kernel", "param_C", "param_gamma"]
    print("Top 10 combinations:")
    print(results[display_cols].head(10).to_string(index=False))

    results_path = RESULTS_DIR / output_name
    results.to_csv(results_path, index=False)
    print(f"[INFO] Saved grid results to {results_path}")

    return grid, results


# === Main workflows ===
def train_multiclass(X, labels, groups, train_idx, test_idx, scaler):
    print("=== MULTICLASS TRAINING ===")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]

    print(f"[INFO] Hold-out test groups: {len(np.unique(test_groups))}")
    print(f"[INFO] CV train groups: {len(np.unique(train_groups))}")

    estimator = SVC(probability=True)
    coarse_grid, coarse_results = run_grid_search(
        "Coarse GridSearch (multiclass)",
        estimator,
        X_train,
        y_train,
        train_groups,
        kernels=["linear", "poly", "rbf", "sigmoid"],
        Cs=COARSE_C_VALUES,
        gammas=COARSE_GAMMA_VALUES,
        output_name="grid_results_coarse_multiclass.csv",
    )

    best_params = coarse_grid.best_params_
    best_kernel = best_params["kernel"]
    best_c = best_params["C"]
    best_gamma = best_params["gamma"]

    print(f"\n[INFO] Best coarse params -> kernel: {best_kernel}, C: {best_c}, gamma: {best_gamma}")

    fine_cs = build_fine_values(best_c, FINE_MULTIPLIERS)
    fine_gammas = build_fine_values(best_gamma, FINE_MULTIPLIERS)

    fine_grid, fine_results = run_grid_search(
        "Fine GridSearch (multiclass)",
        SVC(probability=True),
        X_train,
        y_train,
        train_groups,
        kernels=[best_kernel],
        Cs=fine_cs,
        gammas=fine_gammas,
        output_name="grid_results_fine_multiclass.csv",
    )

    best_model = fine_grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    all_label_indices = np.arange(len(label_encoder.classes_))
    report = classification_report(
        y_test,
        y_pred,
        labels=all_label_indices,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    print("\n=== Evaluation on Hold-out Test (multiclass) ===")
    print(report)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=all_label_indices))

    with open(MODEL_PKL, "wb") as f:
        pickle.dump({
            "model": best_model,
            "label_encoder": label_encoder,
            "finger_cols": LEFT_COLS + RIGHT_COLS,
            "motion_cols": MOTION_COLS,
            "delta_weight": DELTA_WEIGHT,
            "min_delta_mag": MIN_DELTA_MAG,
            "group_column": "base_instance_id",
            "coarse_results": str((RESULTS_DIR / "grid_results_coarse_multiclass.csv").resolve()),
            "fine_results": str((RESULTS_DIR / "grid_results_fine_multiclass.csv").resolve()),
        }, f)

    with open(SCALER_PKL, "wb") as f:
        pickle.dump(scaler, f)

    print("\n[INFO] Multiclass model and scaler have been saved.")

    return label_encoder, y_test, y_pred, best_model


def evaluate_pose_binary(X, labels, groups, train_idx, test_idx, label_encoder):
    print("\n=== PER-POSE ONE-VS-REST EVALUATION ===")
    poses = np.unique(labels)

    X_train, X_test = X[train_idx], X[test_idx]
    train_groups = groups[train_idx]
    test_groups = groups[test_idx]

    summary_rows = []

    for pose in poses:
        print(f"\n--- Pose: {pose} ---")
        y_binary = (labels == pose).astype(int)
        y_train = y_binary[train_idx]
        y_test = y_binary[test_idx]

        positives = y_train.sum()
        negatives = len(y_train) - positives
        if positives == 0 or negatives == 0:
            print("[WARN] Not enough data for binary classification. Skipping.")
            continue

        estimator = SVC(class_weight='balanced', probability=True)
        coarse_grid, _ = run_grid_search(
            f"Coarse GridSearch ({pose} vs rest)",
            estimator,
            X_train,
            y_train,
            train_groups,
            kernels=["linear", "poly", "rbf"],
            Cs=COARSE_C_VALUES,
            gammas=COARSE_GAMMA_VALUES,
            output_name=f"grid_results_coarse_{pose}.csv",
        )

        best_params = coarse_grid.best_params_
        best_kernel = best_params["kernel"]
        best_c = best_params["C"]
        best_gamma = best_params["gamma"]
        print(f"[INFO] Best coarse params -> kernel: {best_kernel}, C: {best_c}, gamma: {best_gamma}")

        fine_cs = build_fine_values(best_c, FINE_MULTIPLIERS)
        fine_gammas = build_fine_values(best_gamma, FINE_MULTIPLIERS)

        fine_grid, _ = run_grid_search(
            f"Fine GridSearch ({pose} vs rest)",
            SVC(class_weight='balanced', probability=True),
            X_train,
            y_train,
            train_groups,
            kernels=[best_kernel],
            Cs=fine_cs,
            gammas=fine_gammas,
            output_name=f"grid_results_fine_{pose}.csv",
        )

        best_model = fine_grid.best_estimator_
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        labels_order = [0, 1]
        target_names = ['other', pose]
        report_dict = classification_report(
            y_test,
            y_pred,
            labels=labels_order,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            y_test,
            y_pred,
            labels=labels_order,
            target_names=target_names,
            zero_division=0,
        )
        print(report_text)
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=labels_order))

        pose_metrics = {
            'pose_label': pose,
            'test_samples': int(len(y_test)),
            'positive_samples': int(y_test.sum()),
            'accuracy': report_dict['accuracy'],
            'precision_pose': report_dict[pose]['precision'],
            'recall_pose': report_dict[pose]['recall'],
            'f1_pose': report_dict[pose]['f1-score'],
        }
        summary_rows.append(pose_metrics)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n=== Summary (per pose) ===")
        print(summary_df.to_string(index=False))
        summary_path = RESULTS_DIR / "pose_binary_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[INFO] Pose summary saved to {summary_path}")


def report_full_dataset(model, label_encoder, X_full, labels_full):
    print("\n=== FULL DATASET EVALUATION ===")
    y_true = label_encoder.transform(labels_full)
    y_pred = model.predict(X_full)
    target_names = label_encoder.classes_
    print(classification_report(y_true, y_pred, target_names=target_names))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nPer-pose accuracy:")
    for idx, pose in enumerate(target_names):
        total = cm[idx].sum()
        correct = cm[idx, idx]
        if total == 0:
            accuracy = 0.0
        else:
            accuracy = correct / total
        print(f"  {pose:15s} total={total:4d} correct={correct:4d} wrong={total - correct:3d} accuracy={accuracy*100:5.1f}%")


# === Main ===
def main(dataset_path: str = DEFAULT_DATASET):
    print("=== TRAIN MOTION SVM WITH FINGER CONTEXT ===")
    print(f"[INFO] Using dataset: {dataset_path}")

    df = load_dataset(dataset_path)
    X, labels, scaler, groups = prepare_features(df)

    train_idx, test_idx = stratified_group_split(labels, groups, test_fraction=TEST_FRACTION, random_state=42)

    label_encoder, y_test_enc, y_pred_enc, best_model = train_multiclass(X, labels, groups, train_idx, test_idx, scaler)
    evaluate_pose_binary(X, labels, groups, train_idx, test_idx, label_encoder)
    report_full_dataset(best_model, label_encoder, X, labels)


def parse_args():
    parser = argparse.ArgumentParser(description="Train motion SVM models with finger context.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to the merged dataset CSV.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dataset_path=args.dataset)
