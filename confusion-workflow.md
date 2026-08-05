# 混淆词实验工作流

日常实验统一通过 `run_pipeline.py` 完成：文本预处理、TTS、音频扰动、WSL ASR、报告
和可选的词典后处理均由一个实验 JSON 控制。

```powershell
python run_pipeline.py --config experiments/configs/full-pipeline-template.json --dry-run
```

输入为 `术语|真实混淆词...` TXT。各阶段可通过 `stages` 开关启停；关闭阶段将复用
`output_dir` 中已有 manifest。

输出布局与当前实验索引见 [outputs/experiments/RESULTS_INDEX.md](outputs/experiments/RESULTS_INDEX.md)。
低层模块 CLI（`error_words_tts.confusion.cli` 等）只用于模块级调试和测试，不再提供
历史根目录包装脚本。
