# Mainline 2: WiSe4Car Annotation Engineering and Results

## Scope

This section consolidates the annotation workflow, toolchain, and current annotation outputs for WiSe4Car.

## A. Annotation Engineering Workflow

Core files:

- WiFi-CSI-Sensing-Benchmark/Annotation_Workspace/video_annotation_tool.py
- WiFi-CSI-Sensing-Benchmark/Annotation_Workspace/csi_annotation_sync.py
- WiFi-CSI-Sensing-Benchmark/Annotation_Workspace/QUICK_START_ANNOTATION.md
- WiFi-CSI-Sensing-Benchmark/Annotation_Workspace/behavior_label_standard.json

End-to-end pipeline (as implemented and documented):

1. Open session video and create action segments.
2. Save per-session annotation JSON.
3. Check video-CSI temporal alignment.
4. Map annotation segments to CSI timeline.
5. Export window-level/labeled training CSV for model training.

Common output forms:

- Session annotation JSON: *_annotations.json
- Labeled windows CSV: labeled_csi_windows_full.csv

## B. Current Annotation Output Snapshot

Main report:

- WiFi-CSI-Sensing-Benchmark/Annotation_Workspace/Annotation_Report.txt

Reported totals:

- analyzed sessions: 75
- total annotations: 297

Action distribution (from Annotation_Report.txt):

- sitting: 129 (43.4%)
- turning: 69 (23.2%)
- reaching: 49 (16.5%)
- using_phone: 34 (11.4%)
- bending: 15 (5.1%)
- waving: 1 (0.3%)

Result directory:

- WiFi-CSI-Sensing-Benchmark/Annotation_Workspace/Annotation_Result

Content includes:

- 75 per-session annotation JSON files
- analysis/export helper scripts:
  - analyze_annotation_results.py
  - export_annotations_csv.py
  - generate_annotation_summary.py

## C. Link to Modeling Data

Annotation-derived labeled file used by downstream preprocessing/training:

- WiFi-CSI-Sensing-Benchmark/Annotation_Workspace/labeled_csi_windows_full.csv

This file is referenced by the WiSe4Car preprocessing route that builds unified numpy tensors used in Transfer_Learning experiments.

## Mainline Conclusion

- Annotation engineering for WiSe4Car is complete as a reproducible pipeline.
- Current labels are imbalanced toward sitting and turning, with very few waving samples.
- Output assets are sufficient for downstream unified dataset generation and target-domain training.
