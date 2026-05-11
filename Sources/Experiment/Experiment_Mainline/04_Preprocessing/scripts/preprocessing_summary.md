# 预处理方法与数据格式总结

## WiSe4Car 预处理（目标域）
- 脚本：WiFi-CSI-Sensing-Benchmark/process_wise4car_to_npy.py
- 数据源：Data/WiSe4Car_Dataset/DataSet_Upload_Version_July_FINAL（原始 CSV）+ Annotation_Workspace/labeled_csi_windows_full.csv（窗口级标注）。
- 主要步骤：
  - 仅取幅度列（pi*_amp_*），按 session 分组、按 rel_time 排序。
  - 窗口长度 4.6s，stride 约 1.0s（由 rel_time 中位差推算步长）。
  - 对每个窗口：重采样时间轴到 T=256；band pooling 到 F=16；去均值/去趋势；z-score（可裁剪到 ±6）。
  - 窗口标签取众数，标签集合固定为 6 类：sitting, reaching, turning, bending, waving, using_phone（输出时首字母大写）。
  - 附加元信息：domain=1（目标域）、session 名、car_id（session 首字符数字）。
- 输出文件（前缀 wise4car_unified_*）：
  - wise4car_unified_features.npy : float32 [N, 256, 16]
  - wise4car_unified_labels.npy   : int64
  - wise4car_unified_domains.npy  : int64（全 1）
  - wise4car_unified_label_names.npy : 长度 6 的字符串数组
  - wise4car_unified_sessions.npy（可选）
  - wise4car_unified_car_ids.npy（可选）

## XRF55 预处理（源域）
- 脚本：XRF55-repo/preprocessing/csi_preprocess.py
- 数据源：dataset/XRF_target/train_list.txt、val_list.txt，对应 train_data/WiFi、test_data/WiFi 下的 .npy（CSI 幅度）。
- 标签映射：原始 act_id → 6 类（Sitting, Reaching, Turning, Bending, Waving, Using Phone），与 WiSe4Car 对齐。
- 主要步骤：
  - 读 .npy，取绝对值；多链路/多 RX 融合为 [time, sub]（mean 或 median）。
  - 可选按采样率中心裁剪到 4.6s；重采样时间轴到 T=256。
  - band pooling 到 F=16（或 stats_pool 可选）；线性去趋势；逐特征 z-score（可裁剪）。
  - 输出 [256, 16]，domain=0。
- 输出文件（目录 XRF55-repo/preprocessed 及拷贝 Transfer_Learning/Data/XRF55）：
  - XRF55_features.npy : float32 [N, 256, 16]
  - XRF55_labels.npy   : int64
  - XRF55_domains.npy  : int64（全 0）
  - XRF55_label_names.npy : 长度 6 的字符串数组
  - XRF55_train_indices.npy / XRF55_val_indices.npy : 预处理时的拆分索引

## 统一数据格式
- 形状：features [N, 256, 16]，float32。
- 标签顺序：6 类统一为 [Sitting, Reaching, Turning, Bending, Waving, Using Phone]。
- 域标识：XRF55=0，WiSe4Car=1（domains.npy）。
- 附加元信息：WiSe4Car 提供 sessions、car_ids；XRF55 当前未附加 session/car。