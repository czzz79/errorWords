# CosyVoice IdeaHub 实验参数矩阵

从现在起，新的解码实验统一使用 CosyVoice 数据；不再将 Qwen3-TTS 音频加入新
实验。历史 108 条混合集 benchmark 仅作留档，不会被新的 JSON 配置引用。

## 共同设置

| 项目 | 设置 |
| --- | --- |
| 术语 | `IdeaHub` |
| GT / 已知混淆词 | `resources/ideahub.txt` |
| 候选混淆词表 | `resources/ideahub_confusion_terms.txt`（32 种写法） |
| CosyVoice 输入 | `outputs/confusion/cosyvoice3/asr-input-manifest.jsonl` + `outputs/confusion/cosyvoice3-gt3-short-six-styles/asr-input-manifest.jsonl` |
| 音频量 | 72 条（每份 manifest 36 条） |
| ASR | Qwen3-ASR，`workers=1`，请求超时 180 秒 |
| 服务 | WSL `Ubuntu-24.04`，端口 `127.0.0.1:8756`，`managed` |
| 随机种子 | `seed=null`，保留服务端随机性 |
| 报告 | 请求/失败数、exact 与 compact match、转写种类数、Top 错误、候选混淆词命中，以及按来源、音色/prompt、扰动分组 |

## 解码实验

| 配置 | 音频 | 条件 | 每条件重复 |
| --- | ---: | --- | ---: |
| `ideahub-temperature-sweep.json` | 72 | `T=0/.15/.35/.55`；固定 `top_p=.95, top_k=50, min_p=.05` | 3 |
| `ideahub-cosy-one-factor.json` | 72 | 基线 `T=.55,p=.95,k=50,min_p=.05`；分别扫描 `top_p=.8/.9/1`、`top_k=0/20/100`、`min_p=0/.02/.1` | 3 |
| `ideahub-cosy-combo.json` | 72 | 基线，及 `(.65,.9,50,.05)`、`(.65,1,50,0)`、`(.8,.9,50,.05)`、`(.8,.95,50,.05)`、`(.8,1,50,0)`、`(1,.95,50,.05)`、`(1,1,50,0)` | 3 |
| `ideahub-cosy-chain.json` | 84 | `greedy=(0,1,0,0)`；`high-random=(1,1,50,0)` | greedy 1；high-random 3 |

四元组顺序为 `(temperature, top_p, top_k, min_p)`。

## 链式扰动实验

链式实验复用已完成的 84 条输入：12 条原始 CosyVoice 音频 + 6 个链式扰动 × 12
条。扰动 seed 为 `20260731`，顺序严格按配置中的 `steps` 执行：

| 名称 | steps |
| --- | --- |
| `speed090-noise18` | speed `.9` → white noise `18 dB` |
| `speed110-noise18` | speed `1.1` → white noise `18 dB` |
| `speed110-lowpass2400` | speed `1.1` → lowpass `2400 Hz` |
| `noise12-lowpass2400` | white noise `12 dB` → lowpass `2400 Hz` |
| `speed110-noise12-lowpass2400` | speed `1.1` → noise `12 dB` → lowpass `2400 Hz` |
| `speed110-noise12-lowpass2400-clip025` | speed `1.1` → noise `12 dB` → lowpass `2400 Hz` → hard clip `.25` |

## 完整管线模板

`full-pipeline-template.json` 的 TTS 运行已经是 CosyVoice：

- TTS 引擎配置：`src/error_words_tts/tts/configs/cosyvoice3-pronunciation.json`
- 发音预处理、原始音频、`noise-18db`、ASR、报告、词典后处理均默认开启。
- 模板的 ASR 服务模式是 `external`；若要由管线启动 WSL 服务，改为 `managed` 并补齐
  `distribution`、`working_directory`、`python` 和 `base_config`。

## GT3 定向文本预处理

`gt3-cosy-pronunciation-preprocess.json` 只运行 pronunciation 阶段：将 1,202 个
GT 词对写入 `resources/gt3-by-type/`，并生成基于 canonical 的中文单音节、英文、
跨语言和数字读法变体。该配置不会加载 CosyVoice，也不会启动 ASR；后续音频实验可
复用其 `samples.json` 并单独开启 TTS 阶段。
