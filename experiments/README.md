# 实验定义

一个实验对应一个 JSON 配置和一个同名 `.ps1`。`.ps1` 顶部仅记录说明，底部只有一条
`python run_pipeline.py --config ...` 命令。

- `configs/full-pipeline-template.json`：新实验模板。
- `configs/ideahub-*.json`：IdeaHub ASR 解码与音频扰动实验定义。
- `configs/gt3-*.json`：GT3 全量和各分类的历史可复现实验定义。

运行结果不写在这里，而是写到配置指定的 `outputs/experiments/<name>/`。该目录的 README
说明实际输入、主要测试内容、结果文件与是否可用于当前统计。
