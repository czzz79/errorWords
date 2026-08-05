# ErrorWords

GT 术语混淆词挖掘的统一实验管线。唯一主入口是 `run_pipeline.py`；实验由
`experiments/configs/*.json` 定义，并有同名 `.ps1` 单命令启动脚本。

## 快速开始

```powershell
cd C:\Users\jsqdc\Desktop\workspace\errorWords
python run_pipeline.py --config experiments/configs/full-pipeline-template.json --dry-run
```

输入 TXT 的一行格式是：

```text
术语|真实混淆词1|真实混淆词2
```

配置中的 `stages` 独立控制 `pronunciation`、`tts`、`augmentation`、`asr`、
`report` 与 `dictionary_postprocess`。关闭某一阶段时，管线会复用同一实验输出目录
中的上游 manifest；缺失时会报错。

## 目录

- `run_pipeline.py`：唯一的实验编排入口，负责 WSL ASR 服务的安全启动和关闭。
- `experiments/`：实验 JSON、单命令 `.ps1` 与说明。
- `resources/`：GT 输入、分类 TXT、词典和文本预处理规则。
- `src/error_words_tts/`：可复用的 TTS、发音预处理、音频扰动、ASR 与报告模块。
- `outputs/experiments/`：每次实验的音频、原始 ASR JSONL、报告和局部 README。

当前实验结果总览见 [outputs/experiments/RESULTS_INDEX.md](outputs/experiments/RESULTS_INDEX.md)。
每个输出实验目录都有 README，说明输入、测试内容和结果位置。
后续未执行的实验设计见 [TODO.md](TODO.md)。

## 运行规则

- `asr.service.mode="managed"`：仅在端口空闲时通过 WSL 启动本次服务，实验结束后仅关闭本次服务。
- `asr.service.mode="external"`：复用外部服务，不会关闭它。
- 不要运行已标记为“中止”的实验目录；它们不纳入任何覆盖率统计。

低层模块 CLI 仍保留在 `src/error_words_tts/`，用于模块测试；日常实验使用
`run_pipeline.py`，不再使用历史根目录运行脚本。
