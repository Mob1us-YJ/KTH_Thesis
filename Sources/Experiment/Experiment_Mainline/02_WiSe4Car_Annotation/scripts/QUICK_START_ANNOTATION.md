# WiSe4Car 视频标注快速开始指南

**创建日期**: 2026-03-04  
**目的**: 快速上手使用视频标注工具

---

## 🚀 5 分钟快速开始

### 第一步：安装依赖

```bash
# 进入项目目录
cd WiFi-CSI-Sensing-Benchmark

# 激活虚拟环境（如果有）
.venv\Scripts\Activate.ps1  # Windows PowerShell

# 安装所需的包
pip install opencv-python pandas
```

### 第二步：选择标注方法

#### 方法 A: 手动标注（推荐第一次尝试）

```bash
python video_annotation_tool.py --video Data/WiSe4Car_Dataset/DataSet_Upload_Version_July_FINAL/101_c1/101_c1.avi
```

**交互式操作**:
- 按 `SPACE` 暂停/播放
- 按方向键快速移动到想要标注的位置
- 按 `1-9` 选择行为类别
- 按 `S` 标记片段开始
- 按 `E` 标记片段结束
- 按 `Q` 退出并保存

#### 方法 B: 带运动检测的半自动标注

```bash
python video_annotation_tool.py --video Data/WiSe4Car_Dataset/DataSet_Upload_Version_July_FINAL/101_c1/101_c1.avi --motion-detect
```

这会先自动检测视频中的运动片段，然后进入手动标注界面。

#### 方法 C: 检查视频-CSI 时间同步

```python
from csi_annotation_sync import VideoCSISyncChecker

# 初始化检查器
video_path = "Data/WiSe4Car_Dataset/DataSet_Upload_Version_July_FINAL/101_c1/101_c1.avi"
csv_dir = "Data/WiSe4Car_Dataset/DataSet_Upload_Version_July_FINAL/101_c1"

checker = VideoCSISyncChecker(video_path, csv_dir)
sync_status = checker.check_sync_status()
```

### 第三步：转换标注为训练数据

```python
from csi_annotation_sync import AnnotationToTrainingData

# 初始化转换器
csv_dir = "Data/WiSe4Car_Dataset/DataSet_Upload_Version_July_FINAL/101_c1"
annotation_json = "101_c1_annotations.json"  # 标注工具生成的文件

converter = AnnotationToTrainingData(csv_dir, annotation_json)

# 加载 CSI 数据
converter.load_csi_data()

# 映射标注到 CSI
converter.map_annotations_to_csi()

# 查看统计
stats = converter.get_statistics()

# 导出训练数据
converter.export_training_data("training_data_101_c1.csv")
```

---

## 📋 行为标注类别速查表

| 按键 | 行为 | 描述 |
|------|------|------|
| **0** | no_motion | 没有人物或完全静止 |
| **1** | sitting | 坐着不动 |
| **2** | reaching | 伸手去够东西 |
| **3** | turning | 转身或转头 |
| **4** | bending | 弯腰或低头 |
| **5** | waving | 挥手或做手势 |
| **6** | talking | 说话或张嘴 |
| **7** | arriving | 进入车辆 |
| **8** | leaving | 离开车辆 |
| **9** | moving_around | 在座位上走动 |

---

## 📊 标注数据格式

标注工具会生成如下格式的 JSON 文件:

```json
{
  "session_id": "101_c1",
  "video_file": "101_c1.avi",
  "video_duration_seconds": 245.3,
  "annotations": [
    {
      "id": 1,
      "start_time": 0.0,
      "end_time": 5.5,
      "start_frame": 0,
      "end_frame": 165,
      "action": "arriving",
      "confidence": 0.95,
      "notes": ""
    },
    {
      "id": 2,
      "start_time": 5.5,
      "end_time": 45.2,
      "action": "sitting",
      "confidence": 1.0,
      "notes": "人物坐在驾驶座上"
    }
  ],
  "annotation_date": "2026-03-04T10:30:00",
  "version": "1.0"
}
```

---

## 🔄 完整的标注-处理-训练流程

### 流程图

```
1️⃣ 播放视频
   ↓ (video_annotation_tool.py)
2️⃣ 手动标注行为段
   ↓ (输出 JSON)
3️⃣ 检查视频-CSI 同步
   ↓ (csi_annotation_sync.py)
4️⃣ 将标注映射到 CSI 数据
   ↓ (按时间戳对齐)
5️⃣ 生成训练数据 CSV
   ↓ (包含标签)
6️⃣ 用于 ML 模型训练
   ↓
7️⃣ 得到行为识别模型 ✅
```

### 完整示例脚本

```python
#!/usr/bin/env python
"""完整的标注→训练数据管道"""

from pathlib import Path
from csi_annotation_sync import AnnotationToTrainingData

# 配置
SESSION_ID = "101_c1"
DATA_ROOT = Path("Data/WiSe4Car_Dataset/DataSet_Upload_Version_July_FINAL")
CSV_DIR = DATA_ROOT / f"{SESSION_ID}"
ANNOTATION_JSON = f"{SESSION_ID}_annotations.json"
OUTPUT_CSV = f"training_data_{SESSION_ID}.csv"

# 执行转换
print(f"处理 {SESSION_ID} ...")
converter = AnnotationToTrainingData(str(CSV_DIR), ANNOTATION_JSON)

print("\n【步骤 1】加载 CSI 数据...")
converter.load_csi_data()

print("\n【步骤 2】映射标注...")
converter.map_annotations_to_csi()

print("\n【步骤 3】统计信息...")
stats = converter.get_statistics()

print("\n【步骤 4】导出训练数据...")
converter.export_training_data(OUTPUT_CSV)

print("\n✅ 完成！可以开始训练模型了。")
```

---

## ⚠️ 常见问题

### Q1: 视频没有声音？
**A**: 这是正常的。标注工具不需要声音，只需要看清楚人物动作。

### Q2: 时间戳对不上怎么办？
**A**: 使用 `VideoCSISyncChecker` 检查同步状态。视频和 CSI 的时间戳应该在几毫秒内对齐。如果偏差很大，可能需要手动调整。

### Q3: 如何处理多人的情况？
**A**: 每个人的行为都单独作为一个标注。如果同时有多个人不同的行为，优先标注主要的人物或主要的行为。

### Q4: 标注时不小心按错了怎么办？
**A**: 按 `C` 键可以撤销最后一个标注。

### Q5: 可以中途保存吗？
**A**: 不行，但退出时会自动保存。建议定期截屏备份重要的标注。

### Q6: 如何加快标注速度？
**A**: 
- 使用方向键快速移动
- 先过一遍视频找出主要的行为段
- 团队中多人并行标注不同的视频

---

## 💡 标注技巧

### 1. 明确的片段划分
✅ **好**: 清晰的开始和结束时间
❌ **不好**: 模糊的边界

### 2. 选择最具体的标签
✅ **好**: `reaching` (伸手去够东西)
❌ **不好**: `moving` (太模糊)

### 3. 避免过度标注
✅ **好**: 只标注清晰可见的重要行为
❌ **不好**: 标注每一个微小的动作

### 4. 记录不确定的情况
如果某个片段不确定，可以：
- 降低 confidence 值（0.5-0.7）
- 在 notes 中说明原因
- 标记为 `unclear` 或 `no_motion`

---

## 📁 文件结构参考

标注完成后的目录结构：

```
WiFi-CSI-Sensing-Benchmark/
├── Data/
│   └── WiSe4Car_Dataset/
│       └── DataSet_Upload_Version_July_FINAL/
│           ├── 101_c1/
│           │   ├── 101_c1_Pi1.csv ← CSI 数据
│           │   ├── 101_c1_Pi2.csv
│           │   ├── ...
│           │   └── 101_c1.avi ← 视频
│           └── ... (其他 session)
│
├── 101_c1_annotations.json ← 标注文件（新生成）
├── training_data_101_c1.csv ← 训练数据（新生成）
└── ... 
```

---

## 🎯 分阶段标注建议

### 第 1 阶段：学习阶段（1-2 个视频）
- 每个视频花 15-30 分钟
- 目的：理解标注规范
- 可标注：`101_c1.avi`, `102_c1.avi` 等单人视频

### 第 2 阶段：规模化标注（10-20 个视频）
- 每个视频花 10-15 分钟（速度变快）
- 目的：建立初步标注数据集
- 并行：最多 3-4 人同时标注不同视频

### 第 3 阶段：质量检查（所有视频）
- 每个视频花 5 分钟复审
- 目的：确保标注一致性
- 方法：随机取 10% 的视频由另一个人重新标注，对比结果

---

## 🔗 相关文件

| 文件 | 功能 |
|-----|------|
| `video_annotation_tool.py` | 交互式标注工具 |
| `csi_annotation_sync.py` | 时间同步和数据转换 |
| `behavior_label_standard.json` | 标注规范定义 |
| `DATASET_ANALYSIS_AND_ANNOTATION_STRATEGY.md` | 详细分析文档 |

---

## 📞 需要帮助？

1. **查看文档**：[DATASET_ANALYSIS_AND_ANNOTATION_STRATEGY.md](DATASET_ANALYSIS_AND_ANNOTATION_STRATEGY.md)
2. **查看标注规范**：[behavior_label_standard.json](behavior_label_standard.json)
3. **查看工具源码**：`video_annotation_tool.py` 的使用方式说明

---

## ✅ 检查清单

在开始标注前，确保：

- [ ] 已安装 OpenCV (`pip install opencv-python`)
- [ ] 已安装 pandas (`pip install pandas`)
- [ ] 能正常打开视频文件
- [ ] 理解了 10 个主要行为类别
- [ ] 已阅读标注规范文档

---

**祝你标注顺利！** 🎉

有任何问题，参考详细分析文档或查看工具的 help 文本。
