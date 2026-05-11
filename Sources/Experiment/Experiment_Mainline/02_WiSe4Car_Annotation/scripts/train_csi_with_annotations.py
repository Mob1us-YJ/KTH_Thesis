"""
Train CSI models using manual video annotations as ground truth.

Workflow:
1) Load aggregated annotations from Annotation_Workspace/all_annotations.csv
2) For each annotated session, aggregate CSI windows from all Pi sensors (100 ms windows)
   using the existing WiSe4Car preprocessing utilities.
3) Align each window to the annotated timeline (relative to session start) and assign action labels.
4) Save the labeled window-level dataset and train a baseline classifier.

This script is lightweight and meant as a starting point. It does not modify any
raw data and only reads CSI CSVs under Data/WiSe4Car_Dataset/DataSet_Upload_Version_July_FINAL/.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Allow importing preprocess_wise4car.py without a package __init__
REPO_ROOT = Path(__file__).resolve().parents[1]
WISE4CAR_ANALYSIS_DIR = REPO_ROOT / "WiSe4Car_Analysis"
if str(WISE4CAR_ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(WISE4CAR_ANALYSIS_DIR))

from preprocess_wise4car import process_session_all_sensors

# Default paths relative to repo root
DEFAULT_DATA_ROOT = REPO_ROOT / "Data" / "WiSe4Car_Dataset" / "DataSet_Upload_Version_July_FINAL"
DEFAULT_ANNOTATION_CSV = REPO_ROOT / "Annotation_Workspace" / "all_annotations.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "Annotation_Workspace" / "labeled_csi_windows.csv"


def load_annotations(csv_path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    """Load annotation intervals per session."""
    df = pd.read_csv(csv_path)
    df['action'] = df['action'].str.strip()
    annotations: Dict[str, List[Tuple[float, float, str]]] = {}
    for session_id, group in df.groupby('session_id'):
        intervals = []
        for _, row in group.iterrows():
            intervals.append((float(row['start_time']), float(row['end_time']), row['action']))
        annotations[session_id] = intervals
    return annotations


def label_session_windows(
    session_name: str,
    session_path: Path,
    intervals: List[Tuple[float, float, str]],
) -> Optional[pd.DataFrame]:
    """Load CSI windows for a session and attach action labels.

    To speed up, we crop windows to the annotated time span (with a small margin).
    """
    if not session_path.exists():
        return None

    features = process_session_all_sensors(session_name, session_path)
    if features is None or features.empty:
        return None

    # Relative seconds from session start
    start_ts = features['timestamp'].min()
    features['rel_time'] = (features['timestamp'] - start_ts).dt.total_seconds()

    # Crop to annotated range to reduce volume
    min_start = min(start for start, _, _ in intervals)
    max_end = max(end for _, end, _ in intervals)
    margin = 1.0  # seconds
    features = features[(features['rel_time'] >= min_start - margin) & (features['rel_time'] <= max_end + margin)]
    if features.empty:
        return None

    def find_label(t: float) -> str:
        for start, end, action in intervals:
            if start <= t <= end:
                return action
        return 'background'

    features['action'] = features['rel_time'].apply(find_label)
    return features


def build_labeled_dataset(
    data_root: Path,
    annotations: Dict[str, List[Tuple[float, float, str]]],
    max_sessions: Optional[int] = None,
    start_index: int = 0,
) -> pd.DataFrame:
    """Aggregate labeled CSI windows for all annotated sessions."""
    labeled_dfs: List[pd.DataFrame] = []
    items = list(annotations.items())
    if start_index:
        items = items[start_index:]
    if max_sessions is not None:
        items = items[:max_sessions]

    for idx, (session_name, intervals) in enumerate(items, 1):
        session_path = data_root / session_name
        print(f"[{idx}/{len(items)}] {session_name} ...", flush=True)
        df = label_session_windows(session_name, session_path, intervals)
        if df is not None:
            df['session'] = session_name
            labeled_dfs.append(df)
        else:
            print(f"Skipping {session_name}: missing data or empty features")

    if not labeled_dfs:
        raise ValueError("No sessions were processed successfully.")

    return pd.concat(labeled_dfs, ignore_index=True)


def train_baseline(df: pd.DataFrame, drop_background: bool = True) -> None:
    """Train a simple RF baseline and print metrics."""
    feature_cols = [c for c in df.columns if c.startswith('pi')]
    data = df.copy()
    if drop_background:
        data = data[data['action'] != 'background']
    data = data.dropna(subset=feature_cols)

    X = data[feature_cols].ffill().fillna(0.0)
    y = data['action']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=24,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    print("Class distribution (train):")
    print(y_train.value_counts())
    print("\nValidation report:")
    print(classification_report(y_val, y_pred, digits=4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CSI model using video annotations")
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="WiSe4Car dataset root")
    parser.add_argument("--annotations", type=str, default=str(DEFAULT_ANNOTATION_CSV), help="all_annotations.csv path")
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_OUTPUT_CSV), help="Where to save labeled windows")
    parser.add_argument("--keep-background", action="store_true", help="Keep background windows during training")
    parser.add_argument("--max-sessions", type=int, default=5, help="Max annotated sessions to process (for quick validation)")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the annotation list for chunked runs")
    parser.add_argument("--append", action="store_true", help="Append to output CSV instead of overwrite")
    parser.add_argument("--skip-train", action="store_true", help="Skip baseline training (useful for chunked runs)")
    parser.add_argument("--train-only", action="store_true", help="Load existing output CSV and only train")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    annotations = load_annotations(Path(args.annotations))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.train_only:
        labeled_df = pd.read_csv(output_path)
        print(f"Loaded existing labeled data from {output_path} (rows={len(labeled_df)})")
    else:
        total = len(annotations)
        use_n = args.max_sessions if args.max_sessions is not None else total - args.start_index
        print(f"Processing {use_n} sessions starting at index {args.start_index} from {data_root} (total annotated={total}) ...")
        labeled_df = build_labeled_dataset(
            data_root,
            annotations,
            max_sessions=args.max_sessions,
            start_index=args.start_index,
        )

        write_mode = 'a' if args.append and output_path.exists() else 'w'
        header = not (args.append and output_path.exists())
        labeled_df.to_csv(output_path, index=False, mode=write_mode, header=header)
        print(f"Saved labeled windows to {output_path} (rows added={len(labeled_df)})")

        if args.skip_train:
            return

    train_baseline(labeled_df, drop_background=not args.keep_background)


if __name__ == "__main__":
    main()
