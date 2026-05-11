# Mainline 1: Benchmark on UT-HAR and NTU-Fi_HAR

## Scope

This section tracks benchmark comparison results focused on two datasets:

- UT-HAR
- NTU-Fi_HAR

## A. UT-HAR Benchmark (supervised)

Primary script and fixed comparison data:

- WiFi-CSI-Sensing-Benchmark/BenchTest/compare_models.py

Recorded test accuracy from the script:

- LeNet (CNN): 97.46%
- CNN+GRU: 97.13%
- MLP: 91.64%
- BiLSTM (fixed version): 77.50%
- RNN: 75.87%

Recorded train-test gap from the script:

- CNN+GRU: 2.29%
- LeNet: 2.54%
- MLP: 3.90%
- BiLSTM: 4.00%
- RNN: 6.74%

UT-HAR output figure in BenchTest:

- WiFi-CSI-Sensing-Benchmark/BenchTest/model_comparison.png

## B. NTU-Fi_HAR Benchmark

### B1. Supervised baseline on NTU-Fi_HAR

Main driver:

- WiFi-CSI-Sensing-Benchmark/BenchTest/compare_models_ntufi.py

Result file:

- WiFi-CSI-Sensing-Benchmark/BenchTest/ntufi_results.json

Test accuracy (from ntufi_results.json):

- BiLSTM: 100.00%
- CNN+GRU: 99.62%
- MLP: 98.11%
- LeNet: 97.35%
- RNN: 85.23%

### B2. Transfer benchmark linked to NTU-Fi_HAR

Summary document:

- WiFi-CSI-Sensing-Benchmark/BenchTest/report.md

Main setup documented there:

- Self-supervised pretraining data: NTU-Fi_HAR train_amp + test_amp.
- Linear evaluation supervised data: NTU-Fi-HumanID.
- Representative run settings: self-epochs=100, super-epochs=300.

Key plotting/summary scripts:

- WiFi-CSI-Sensing-Benchmark/BenchTest/summary_pre_experiment.py
- WiFi-CSI-Sensing-Benchmark/BenchTest/plot_self_supervised_results.py

Representative output artifacts:

- WiFi-CSI-Sensing-Benchmark/BenchTest/pre_experiment_summary.png
- WiFi-CSI-Sensing-Benchmark/BenchTest/ntufi_model_comparison.png
- WiFi-CSI-Sensing-Benchmark/BenchTest/quick_results/results_*.npz

## Mainline Conclusion

- UT-HAR: LeNet and CNN+GRU are the strongest among listed models, with low generalization gap.
- NTU-Fi_HAR supervised benchmark: BiLSTM and CNN+GRU perform best.
- NTU-Fi_HAR also serves as the pretraining source in transfer experiments documented in report.md.
