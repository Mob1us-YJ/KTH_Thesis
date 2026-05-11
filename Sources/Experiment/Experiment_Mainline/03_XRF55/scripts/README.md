# XRF55 搜索脚本使用要点

- **分批跑 Stage A**：
  - 限定模型子集：`--models plain resnet`。
  - 跳过已完成（检测 metrics.json）：默认开启，可用 `--no_skip_existing` 关闭。
  - 示例：
    ```bash
    python -m Transfer_Learning.Training.xrf55_framework.search.run_stage_a \
      --data_dir Transfer_Learning/Data/XRF55 --epochs 80 --batch_size 128 --seed 42 \
      --models plain resnet
    ```

- **Stage B 随机搜索**：
  - 已完成的 trial（按 model+trial 前缀找 metrics.json）会被跳过，可用 `--no_skip_existing` 关闭。
  - 示例：
    ```bash
    python -m Transfer_Learning.Training.xrf55_framework.search.run_stage_b \
      --data_dir Transfer_Learning/Data/XRF55 --models plain resnet multiscale \
      --trials 5 --seed 42
    ```

- **Stage C 稳健性**：
  - 已完成的 (model, seed) 会跳过，可用 `--no_skip_existing` 关闭。
  - 示例：
    ```bash
    python -m Transfer_Learning.Training.xrf55_framework.search.run_stage_c \
      --data_dir Transfer_Learning/Data/XRF55 \
      --configs Transfer_Learning/Training/xrf55_framework/search/configs_stage_c_sample.json \
      --seeds 1 2 3
    ```

- **日志位置**：`Transfer_Learning/result/xrf55_framework/<run_name>/`，包含 config、metrics、history、曲线、混淆矩阵、best.pt。

- **终止/续跑策略**：
  - Stage A/B/C 都可多次调用；默认跳过已完成 run，方便分段运行或中途中断后续跑。
  - 如需强制重跑，加入 `--no_skip_existing`。
