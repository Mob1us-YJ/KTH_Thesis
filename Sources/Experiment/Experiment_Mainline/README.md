# Experiment Mainline Overview

Updated: 2026-05-04

This folder consolidates the four core experiment threads in this repository:

1. Benchmark comparisons on UT-HAR and NTU-Fi_HAR.
2. WiSe4Car annotation engineering workflow and annotation outputs.
3. XRF55 preprocessing pipeline, pretraining/backbone selection, and hyperparameter search results.
4. CSI representation visualization — demonstrating 1D, STFT, and ACF representations with interactive plots.

## Folder Contents

- 01_Benchmark_UT_HAR_NTU_FI_HAR.md
- 02_WiSe4Car_Annotation_Mainline.md
- 03_XRF55_Preprocess_Pretrain_HParam_Mainline.md
- 04_Preprocessing/
  - README.md — Quick-start guide with performance analysis
  - 00_MODULE_SUMMARY.md — Module overview
  - 01_WHY_ACF_BEST_ANALYSIS.md — **NEW: Detailed visual proof of ACF > STFT > 1D**
  - 03_CSI_Representations_Guide.md — Technical specifications
  - scripts/
    - visualize_csi_representations.py — 3-panel CSI representation comparison
    - compare_representations_performance.py — **NEW: 5-panel why-ACF-best analysis**
    - preprocess_stft_acf.py, preprocessing_summary.md, stft_acf_preprocessing_design.md (copied)
  - results/
    - csi_representations_*.png/pdf — Basic 3-panel comparisons
    - per_band_analysis_*.png/pdf — Detailed per-band analysis
    - why_acf_best_Waving.png/pdf — **NEW: 4-sample Waving action comparison**
    - why_acf_best_Turning.png/pdf — **NEW: 3-sample Turning action comparison**

## Local Bundles (copied scripts/results)

- 01_Benchmark/
	- scripts/: benchmark scripts and report
	- results/: benchmark result json/png + quick_results/
- 02_WiSe4Car_Annotation/
	- scripts/: annotation toolchain scripts
	- results/: annotation report, full labeled csv, Annotation_Result/
- 03_XRF55/
	- scripts/: XRF55 preprocessing/training/search scripts
	- results/: preprocessing docs and stage summary results

## Source Anchors (original files)

- WiFi-CSI-Sensing-Benchmark/BenchTest/report.md
- WiFi-CSI-Sensing-Benchmark/BenchTest/ntufi_results.json
- WiFi-CSI-Sensing-Benchmark/BenchTest/compare_models.py
- WiFi-CSI-Sensing-Benchmark/Annotation_Workspace/Annotation_Report.txt
- WiFi-CSI-Sensing-Benchmark/Annotation_Workspace/QUICK_START_ANNOTATION.md
- Transfer_Learning/result/preprocessing_summary.md
- Transfer_Learning/result/stft_acf_preprocessing_design.md
- Transfer_Learning/result/xrf55_training_summary.md
- Transfer_Learning/result/xrf55_framework/stage_a_summary.csv
- Transfer_Learning/result/xrf55_framework/stage_b_summary.csv
- XRF55-repo/preprocessing/csi_preprocess.py
- XRF55-repo/preprocessing/train_xrf55_encoder.py
- Transfer_Learning/Training/xrf55_framework/search/run_stage_a.py
- Transfer_Learning/Training/xrf55_framework/search/run_stage_b.py
- Transfer_Learning/Training/xrf55_framework/search/search_space.py
