# ErrorWords 文件格式与 Pipeline 指南

本文说明统一入口、各阶段输入输出的基本格式，以及如何定位一次实验的结果。

## 1. 统一入口

日常实验使用项目根目录的 `run_pipeline.py`，实验参数放在
`experiments/configs/*.json`：

```powershell
uv run python .\run_pipeline.py `
  --config .\experiments\configs\gt3-english-cmu-cosy-intranet-augment-asr.json `
  --dry-run

uv run python .\run_pipeline.py `
  --config .\experiments\configs\gt3-english-cmu-cosy-intranet-augment-asr.json
```

`--dry-run` 只解析配置、输入和阶段依赖，不生成音频、不调用 ASR。也可以用
`--stages pronunciation,tts,augmentation,asr,report` 临时覆盖配置中的阶段开关。

## 2. 输入文件

### 2.1 GT TXT

`input_txt` 指向 UTF-8 文本，每行一个术语及其已知混淆词：

```text
IdeaHub|ID Hub|Idea hot|Idea Hub
周清|周星
```

- 第一个字段是 canonical/GT 术语。
- 后续字段是该术语的真实混淆词候选。
- 空行和 `#` 开头的行会忽略。
- 相同 canonical 会合并，候选会去重。

### 2.2 实验 JSON

实验 JSON 至少包含：

```json
{
  "input_txt": "resources/gt3-by-type/english-word-acronym.txt",
  "output_dir": "outputs/experiments/example",
  "stages": {
    "pronunciation": true,
    "tts": true,
    "augmentation": true,
    "asr": true,
    "report": true,
    "dictionary_postprocess": false
  }
}
```

TTS、扰动和 ASR 的具体参数均放在同一个实验 JSON 中；模型路径、音色和 CosyVoice
输入模式仍由引用的 engine JSON 控制。

### 2.3 外部 clean TTS manifest

如果关闭 pronunciation/TTS，实验可以复用已有 manifest。例如内网续跑配置使用：

```text
inputs/generatedaudios/gt3-english-cmu-cosy-clean/tts-manifest.jsonl
```

该文件是 JSONL，每行一个音频样本。最重要的字段如下：

```json
{
  "sample_id": "gt-0001-...",
  "text": "API key",
  "audio_path": ".../sample.wav",
  "status": "generated",
  "phoneme_text": "[EY1][P][IY1][AY1] [K][IY1]",
  "input_mode": "phoneme",
  "target_confusions": ["APIK"]
}
```

`audio_path` 必须能从当前机器访问；复制或解压音频时要保留 manifest 能解析的目录布局。

## 3. 各阶段输出格式

所有 JSONL 都是“每行一个独立 JSON 对象”，可以边运行边追加和读取。

### 3.1 pronunciation：`samples.json` 和 `pronunciation/`

`samples.json` 是 JSON 数组，保存送入 TTS 的样本。常见字段包括：

- `id`、`text`、`canonical_text`：样本身份和 GT 术语；
- `tts_text`、`source_text`、`text_source`：文本来源；
- `phoneme_text`、`input_mode`：CosyVoice 音素输入及模式；
- `pronunciation_rule`、`variant_kind`、`pronunciation_delta`：变体审计信息；
- `target_confusions`、`confusion_category`：对应 GT 混淆词和类别。

英文 CMU 预处理还会产生：

```text
pronunciation/pronunciation-variants.jsonl
pronunciation/structured-pronunciations.jsonl
pronunciation/summary.json
pronunciation/unresolved.jsonl
```

其中 `pronunciation-variants.jsonl` 每行一个 canonical 或扰动变体，
`phoneme_text` 形如 `[AY1][D] [HH][AH1][B]`；`unresolved.jsonl` 记录无法可靠生成读法的术语。

### 3.2 TTS：`tts/manifest.jsonl` 和 WAV

TTS manifest 在样本字段之外增加：

```json
{
  "engine": "cosyvoice3",
  "audio_path": ".../audio.wav",
  "status": "generated",
  "sample_rate": 16000,
  "duration_ms": 1920,
  "metadata": {
    "input_mode": "phoneme",
    "synthesis_text": "[EY1][P][IY1][AY1] [K][IY1]"
  }
}
```

`status` 通常为 `generated`、`cached` 或错误状态。只有 `generated/cached` 样本会进入后续 ASR。

### 3.3 augmentation：`augmentation/manifest.jsonl`

扰动配置记录本次使用的 seed 和操作列表；manifest 仍是一行一个样本，但会更新：

```json
{
  "audio_path": ".../augmentation/speed-090/sample.wav",
  "source_audio_path": ".../tts/sample.wav",
  "augmentation": {
    "name": "speed-090",
    "parameters": {"factor": 0.9}
  },
  "status": "generated"
}
```

canonical 元数据（`target_confusions`、`variant_kind`、`phoneme_text` 等）会沿 manifest 传递。

### 3.4 ASR 输入：`asr-input-manifest.jsonl`

这是 TTS clean manifest 与扰动 manifest 的合并结果。ASR 客户端只要求每行至少有：

```json
{
  "sample_id": "...",
  "text": "API key",
  "audio_path": "...wav",
  "status": "generated"
}
```

其它发音、扰动和 GT 字段会原样保留，便于报告按术语、规则、音频扰动分组。

### 3.5 ASR 结果：`results/<condition>/run-XX.jsonl`

每行包含原始样本信息，以及：

```json
{
  "asr": {
    "status": "success",
    "text": "A P Y key.",
    "raw_text": "language English<asr_text>A P Y key.",
    "service_url": "http://.../v1/audio/transcriptions",
    "backend": "openai_http",
    "http_status": 200
  },
  "comparison": {
    "exact_match": false,
    "compact_match": false
  }
}
```

内网服务返回的 `language English<asr_text>` 包装会从 `text` 中清除，原始值保存在 `raw_text`。
每个解码条件和重复运行使用独立的 `run-01.jsonl`、`run-02.jsonl`；旁边的
`.meta.json` 保存输入指纹和解码设置，避免误复用旧结果。

### 3.6 报告和混淆词输出

`report/summary.json` 是完整汇总，包含请求数、失败数、exact/compact match、不同转写种类、
候选命中和 GT 混淆词命中。常用 CSV：

```text
report/summary.csv
report/group-breakdown.csv
report/ground-truth-confusion-hits.csv
report/ground-truth-confusion-misses.csv
report/cmu-rule-effectiveness.csv
```

`confusion-words.txt` 使用和 GT 相同的格式：

```text
IdeaHub|ID Hub|Idea hot
```

### 3.7 ASR 前音频预处理：`asr_preprocess`

`asr_preprocess` 是 `augmentation` 和 `asr` 之间的独立可选阶段。它不访问 ASR 服务，
只把上游 manifest 中的音频校验为 16 kHz、单声道、PCM16 WAV，并按能量 VAD 切出语音片段。
输出位于：

```text
asr-preprocess/audio/<row>-<sample-id>/segment-0000.wav
asr-preprocess/manifest.jsonl
asr-preprocess/summary.json
```

示例配置：

```json
{
  "stages": {
    "augmentation": true,
    "asr_preprocess": true,
    "asr": true
  },
  "asr_preprocess": {
    "enabled": true,
    "use_vad": true,
    "threshold": 0.02,
    "frame_ms": 20,
    "padding_ms": 200,
    "silence_finalize_ms": 600,
    "min_speech_ms": 250,
    "merge_gap_ms": 300
  }
}
```

关闭 `asr_preprocess` 时，ASR 使用原来的 `asr-input-manifest.jsonl`；开启后 ASR 自动改读
`asr-preprocess/manifest.jsonl`。只生成预处理文件而不启动 ASR 时，可运行：

```powershell
uv run python .\run_pipeline.py --config .\experiments\configs\your-experiment.json --stages asr_preprocess
```

如果开启 `dictionary_postprocess`，还会生成带词典标记、删除项和 review JSONL 的文件；这些是
后处理结果，不应覆盖原始 ASR JSONL。

## 4. Pipeline 执行顺序

```text
配置 JSON + GT TXT
        |
        v
1. pronunciation  文本/音素预处理，生成 samples.json
        |
        v
2. tts            调用 CosyVoice/Qwen3-TTS，生成 WAV + tts/manifest.jsonl
        |
        v
3. augmentation   对 clean WAV 做 speed/noise/lowpass 等扰动
        |
        v
4. manifest merge 合并 clean 与扰动音频为 asr-input-manifest.jsonl
        |
        v
5. asr            managed 启 WSL 或 external/openai_http 复用服务
        |
        v
6. report         统计识别、候选混淆词和 GT 混淆词覆盖率
        |
        v
7. dictionary_postprocess（可选）对 confusion-words.txt 做词典过滤
```

阶段关闭时的复用规则：

- 关闭 pronunciation 但开启 TTS：必须已有 `samples.json`；
- 关闭 TTS 但开启 augmentation：必须已有 TTS manifest，或配置 `augmentation.input_manifest`；
- 关闭 augmentation 但开启 ASR：必须已有 `asr-input-manifest.jsonl`，或配置 `asr.input_manifests`；
- 关闭 ASR 但开启 report/postprocess：必须已有 `results/*/run-*.jsonl`。

缺少上游产物会直接报错，不会静默重新生成或使用其它实验目录。

## 5. ASR 服务入口

服务有两种模式：

- `local_wsl`：本机 WSL Qwen3-ASR，通常是 `127.0.0.1:8756`，可由 Python 管理启停；
- `openai_http`：外部 HTTP multipart 服务，例如内网
  `http://200.4.188.200:18000/v1/audio/transcriptions`，配置为 `service.mode=external`，不会被 pipeline 停止。

API key 只从 `QWEN_ASR_API_KEY` 环境变量读取，不写进配置或 Git。

## 6. 单个音频快速测试

当前没有单文件评分版 ASR pipeline；批量客户端 `src/error_words_tts/asr_cli.py` 的正式入口需要
manifest。但仓库已有单音频探针：

```powershell
$env:QWEN_ASR_API_KEY = "真实 key"
uv run python .\tools\probe_asr_endpoint.py `
  --base-url "http://200.4.188.200:18000/v1" `
  --audio .\test.wav `
  --model "Qwen3-ASR"
```

它会检查 TCP、`/models` 和 `/audio/transcriptions`，并打印服务返回的原始 JSON；它不写实验结果，
适合接口连通性和单个音频试听前的快速验证。
