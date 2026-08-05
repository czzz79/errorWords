# 实验入口

每个可运行实验由下列两项组成：

- `experiments/configs/<name>.json`：输入、阶段开关、TTS、扰动、ASR 解码参数和输出目录。
- `experiments/<name>.ps1`：说明注释加一条 `python run_pipeline.py --config ...` 命令。

运行前验证：

```powershell
python run_pipeline.py --config experiments/configs/<name>.json --dry-run
```

运行实验：

```powershell
.\experiments\<name>.ps1
```

结果写入 JSON 中的 `output_dir`。完整的已完成实验、当前覆盖率和每个输出文件的
含义见 [outputs/experiments/RESULTS_INDEX.md](outputs/experiments/RESULTS_INDEX.md)。

`full-pipeline-template.json` 是新实验的模板；IdeaHub 的温度、单因素、联合采样和
链式扰动配置保留用于 ASR 解码回归测试。GT3 的历史配置保留以保证结果可复现。
