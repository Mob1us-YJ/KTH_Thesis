# Mainline 3: XRF55 Preprocessing, Pretraining, and Hyperparameter Results

## Scope

This section consolidates:

- XRF55 data preprocessing path and unified format.
- Pretraining/backbone and trial search framework.
- Existing performance outcomes and selected configurations.

## A. Preprocessing Pipeline

Core implementation and design documents:

- XRF55-repo/preprocessing/csi_preprocess.py
- Transfer_Learning/result/preprocessing_summary.md
- Transfer_Learning/preprocess_stft_acf.py
- Transfer_Learning/result/stft_acf_preprocessing_design.md

Unified output format:

- features shape: [N, 256, 16]
- labels: 6 classes aligned with WiSe4Car
- domains: XRF55 uses domain_id=0

Saved files (XRF55 side):

- XRF55_features.npy
- XRF55_labels.npy
- XRF55_domains.npy
- XRF55_label_names.npy
- XRF55_train_indices.npy
- XRF55_val_indices.npy

Representation extension route:

- STFT tensors and ACF tensors are generated from unified 1D features.
- STFT output per sample: [16, 65, 13]
- ACF output per sample: [1, 65, 16]

## B. Pretraining Encoder and Training Hyperparameters

Main training script:

- XRF55-repo/preprocessing/train_xrf55_encoder.py

Model and training controls in script:

- 1D Conv encoder width ladder (width, 2w, 4w, 8w)
- dropout, label smoothing, mixup_alpha
- optimizer: Adam
- scheduler: step or cosine
- weighted sampler optional
- class-weighted cross entropy
- early stopping with patience

Documented run summary:

- Transfer_Learning/result/xrf55_training_summary.md

Best run from that summary:

- preproc_cosine_wide
- width=96, cosine scheduler, no sampler
- best validation accuracy about 0.8552

## C. Framework-Level Backbone and Hyperparameter Search

Search framework entry points:

- Transfer_Learning/Training/xrf55_framework/search/run_stage_a.py
- Transfer_Learning/Training/xrf55_framework/search/run_stage_b.py
- Transfer_Learning/Training/xrf55_framework/search/search_space.py
- Transfer_Learning/Training/xrf55_framework/search/README.md

Stage A summary file:

- Transfer_Learning/result/xrf55_framework/stage_a_summary.csv

Observed Stage A best (from summary):

- model: plain
- base_width: 96, num_blocks: 4
- best_val_acc: 0.8552188552

Stage B summary file:

- Transfer_Learning/result/xrf55_framework/stage_b_summary.csv

Observed Stage B best by model family (from visible rows):

- plain best: 0.8989898990
- resnet best: 0.9276094276
- multiscale best: 0.9343434343

Current top visible Stage B run:

- multiscale trial1 with best_val_acc 0.9343434343

## D. Result Artifacts

Main result roots:

- Transfer_Learning/result/xrf55_framework/
- XRF55-repo/result/

Common per-run artifacts:

- config.json
- metrics.json
- history.csv
- confusion_matrix.png
- curves.png
- best.pt or encoder_best.pth

## Mainline Conclusion

- XRF55 preprocessing is already aligned to WiSe4Car in temporal length, feature width, and label space.
- Single-script encoder runs reached about 0.8552 validation accuracy (best documented in xrf55_training_summary.md).
- Framework search (Stage B) shows stronger candidates, with multiscale currently highest among listed results.
