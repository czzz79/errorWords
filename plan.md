# Qwen3-ASR 专业术语混淆词挖掘 Pipeline 实施计划

## 1. 项目背景

当前实时会议语音识别系统使用 `Qwen3-ASR-1.7B` 进行流式或离线解码。对于产品名、模型名、行业术语、人名、缩写和中英文混合术语，通用 ASR 容易出现以下问题：

- 音近字替换；
- 专有名词被替换为常见词；
- 英文缩写被错误拆分、合并或音译；
- 术语部分丢失；
- 流式 partial 阶段出现错误，final 阶段才恢复；
- 在噪声、混响、语速变化和口音条件下错误率显著升高；
- 简单全局替换会把正常常见词误改成术语。

本项目拟采用低成本方案，通过合成数据主动探测 `Qwen3-ASR-1.7B` 对目标术语的真实错误分布，构建：

1. 术语实际混淆词库存；
2. 可直接替换的安全别名表；
3. 需要上下文判定的混淆规则；
4. 防止误替换的 hard negative 数据；
5. 面向后续纠错模型或规则系统的训练与评测数据。

---

## 2. 项目目标

### 2.1 核心目标

构建以下自动化流水线：

```text
术语库
  ↓
LLM 生成多样化上下文
  ↓
多 TTS、多音色、多发音形式生成音频
  ↓
音频增强：语速、噪声、混响、设备和编码变化
  ↓
TTS 质量过滤
  ↓
Qwen3-ASR-1.7B 解码
  ↓
标准文本与 ASR 输出对齐
  ↓
抽取术语对应错误片段
  ↓
聚合、打标签、风险分类
  ↓
人工审核 Top-K
  ↓
安全别名 / 上下文规则 / Hard Negative
```

### 2.2 第一阶段不做的事情

MVP 阶段暂不进行：

- Qwen3-ASR 参数微调；
- 大规模声学模型训练；
- 自定义 beam search 或热词解码器开发；
- 全自动替换线上 ASR 输出；
- 依赖单个 LLM 直接生成术语别名；
- 数十万术语的全量生成。

MVP 的目标是先验证：

> 合成音频能否稳定复现真实 ASR 术语错误，以及这些错误能否被结构化为可安全使用的混淆知识。

---

## 3. 核心设计原则

### 3.1 LLM 不直接决定最终别名

LLM 可以负责：

- 生成术语上下文；
- 生成口语化表达；
- 生成 hard negative 句子；
- 对实际 ASR 输出进行错误类型标注；
- 辅助判断是否需要上下文；
- 对人工审核样本排序。

最终混淆词必须来自：

```text
真实生成音频 → Qwen3-ASR-1.7B 实际解码结果
```

不能仅根据 LLM 的语言常识推测。

### 3.2 区分术语读法与标准写法

每个术语必须同时保存：

- 标准展示形式；
- 合法别名；
- 实际口语读法；
- 拼音或音素；
- 中英文混读形式；
- 可能的缩写展开方式。

例如：

```json
{
  "canonical": "Qwen3-ASR",
  "spoken_forms": [
    "Qwen three A S R",
    "Qwen 三 A S R",
    "千问三语音识别"
  ],
  "display_aliases": [
    "Qwen3 ASR",
    "Qwen ASR"
  ]
}
```

必须避免把 TTS 的错误发音误当作 ASR 的错误。

### 3.3 不跑完整笛卡尔积

音色、上下文、语速、噪声、混响和 TTS 引擎的完整组合会快速膨胀。

采用：

- 分层随机采样；
- 拉丁超立方或近似均衡采样；
- 对高风险术语追加样本；
- 对稳定正确术语减少采样。

### 3.4 原始结果必须保留

所有纠错和标签都必须保留：

- 原始标准文本；
- TTS 输入文本；
- 实际 spoken form；
- 音频增强参数；
- ASR partial；
- ASR final；
- 对齐结果；
- 自动标签；
- 人工标签。

禁止只保留最终别名表，否则无法追溯错误来源。

---

## 4. MVP 范围

### 4.1 术语数量

首轮选取：

- 30～50 个真实业务术语；
- 覆盖中文、英文缩写、中英文混合、人名、产品名、数字型号；
- 每类至少 5 个术语。

建议类别：

| 类别 | 示例 |
|---|---|
| 中文行业术语 | 外板纵骨、舷侧纵桁 |
| 英文缩写 | ASR、VAD、NPU、RNN-T |
| 中英文混合 | Qwen3-ASR、Ascend 310P |
| 产品或模型名 | CosyVoice、WebSocket |
| 人名或项目名 | 真实会议常见人名、项目名 |
| 数字型号 | 310P、1.7B、16kHz |

### 4.2 样本数量

每个术语初始生成约 150～300 条音频。

建议首轮总量：

```text
40 个术语 × 200 条 ≈ 8,000 条音频
```

总量控制在 5,000～12,000 条之间。

### 4.3 TTS 引擎

首轮至少使用两个不同 TTS 来源：

1. `CosyVoice 3`
2. 以下任选一个：
   - Azure Speech SSML
   - OpenVoice V2
   - GPT-SoVITS

推荐：

```text
CosyVoice 3：70%
第二路 TTS：30%
```

### 4.4 ASR 解码模式

必须使用与生产环境一致的配置：

- `Qwen3-ASR-1.7B`
- 实际 checkpoint；
- 实际流式 chunk；
- 实际 VAD；
- 实际 finalization；
- 实际文本后处理；
- 实际解码参数。

同时保留：

- offline final；
- streaming partial 序列；
- streaming final。

---

## 5. 目录结构

```text
term-confusion-mining/
├── README.md
├── plan.md
├── config/
│   ├── pipeline.yaml
│   ├── tts.yaml
│   ├── augmentation.yaml
│   ├── asr.yaml
│   └── llm.yaml
├── data/
│   ├── terms/
│   │   ├── terms.jsonl
│   │   └── pronunciation_lexicon.jsonl
│   ├── contexts/
│   │   ├── generated_contexts.jsonl
│   │   └── hard_negative_contexts.jsonl
│   ├── manifests/
│   │   ├── tts_manifest.jsonl
│   │   ├── augmentation_manifest.jsonl
│   │   ├── asr_manifest.jsonl
│   │   └── aligned_manifest.jsonl
│   ├── audio/
│   │   ├── clean/
│   │   └── augmented/
│   ├── asr_outputs/
│   ├── alignments/
│   ├── confusion_inventory/
│   ├── reviewed/
│   └── reports/
├── src/
│   ├── term_schema.py
│   ├── context_generator.py
│   ├── tts/
│   │   ├── base.py
│   │   ├── cosyvoice_backend.py
│   │   └── secondary_backend.py
│   ├── augmentation/
│   │   ├── pipeline.py
│   │   └── sampling.py
│   ├── quality/
│   │   ├── tts_validator.py
│   │   └── audio_validator.py
│   ├── asr/
│   │   ├── qwen3_offline.py
│   │   └── qwen3_streaming.py
│   ├── alignment/
│   │   ├── text_alignment.py
│   │   ├── pinyin_alignment.py
│   │   └── term_span_extractor.py
│   ├── mining/
│   │   ├── confusion_aggregator.py
│   │   ├── risk_classifier.py
│   │   └── candidate_ranker.py
│   ├── labeling/
│   │   ├── rule_labeler.py
│   │   └── llm_labeler.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── report.py
│   └── cli.py
└── scripts/
    ├── 01_generate_contexts.py
    ├── 02_generate_tts.py
    ├── 03_augment_audio.py
    ├── 04_run_asr.py
    ├── 05_align_and_extract.py
    ├── 06_aggregate_confusions.py
    ├── 07_llm_tagging.py
    ├── 08_export_review_sheet.py
    └── 09_build_alias_tables.py
```

---

## 6. 数据结构设计

## 6.1 术语表 `terms.jsonl`

```json
{
  "term_id": "ship_0001",
  "canonical": "外板纵骨",
  "language": "zh",
  "domain": "ship_structure",
  "category": "technical_term",
  "priority": 8,
  "spoken_forms": [
    "外板纵骨"
  ],
  "display_aliases": [],
  "pinyin": "wai ban zong gu",
  "phonemes": null,
  "similar_terms": [
    "外板纵桁",
    "舷侧纵骨"
  ],
  "context_keywords": [
    "间距",
    "布置",
    "船体",
    "肋位"
  ],
  "notes": ""
}
```

## 6.2 上下文表 `generated_contexts.jsonl`

```json
{
  "context_id": "ctx_ship_0001_0001",
  "term_id": "ship_0001",
  "text": "我们先调整外板纵骨的布置方案。",
  "spoken_text": "我们先调整外板纵骨的布置方案。",
  "term_span": [6, 10],
  "term_position": "middle",
  "sentence_type": "declarative",
  "register": "meeting_spoken",
  "contains_negation": false,
  "contains_similar_term": false,
  "contains_number": false,
  "context_source": "llm",
  "llm_prompt_version": "v1"
}
```

## 6.3 TTS Manifest

```json
{
  "sample_id": "sample_000001",
  "term_id": "ship_0001",
  "context_id": "ctx_ship_0001_0001",
  "tts_engine": "cosyvoice3",
  "tts_model_version": "2026-xx",
  "speaker_id": "speaker_03",
  "reference_audio_id": "ref_03",
  "spoken_form": "外板纵骨",
  "speed_instruction": "normal",
  "emotion": "neutral",
  "accent": "mandarin",
  "seed": 12345,
  "clean_audio_path": "data/audio/clean/sample_000001.wav",
  "sample_rate": 16000,
  "duration_ms": 2350
}
```

## 6.4 增强 Manifest

```json
{
  "sample_id": "sample_000001_aug_02",
  "parent_sample_id": "sample_000001",
  "audio_path": "data/audio/augmented/sample_000001_aug_02.wav",
  "speed_factor": 1.08,
  "pitch_shift_semitones": 0.0,
  "noise_type": "meeting_room",
  "snr_db": 10,
  "rir_id": "small_room_04",
  "codec": "opus",
  "lowpass_hz": 7200,
  "gain_db": -1.5,
  "augmentation_seed": 9081
}
```

## 6.5 ASR 输出

```json
{
  "sample_id": "sample_000001_aug_02",
  "asr_model": "Qwen3-ASR-1.7B",
  "decode_mode": "streaming",
  "decode_config_hash": "abc123",
  "partial_hypotheses": [
    {
      "timestamp_ms": 820,
      "text": "我们先调整外板"
    },
    {
      "timestamp_ms": 1130,
      "text": "我们先调整外版中国"
    },
    {
      "timestamp_ms": 1560,
      "text": "我们先调整外板纵骨"
    }
  ],
  "final_text": "我们先调整外板纵骨的布置方案",
  "decode_latency_ms": 1650
}
```

## 6.6 混淆记录

```json
{
  "term_id": "ship_0001",
  "canonical": "外板纵骨",
  "reference_span": "外板纵骨",
  "hypothesis_span": "外版中国",
  "normalized_hypothesis_span": "外版中国",
  "error_type": "phonetic_substitution",
  "decode_stage": "partial",
  "sample_id": "sample_000001_aug_02",
  "speaker_id": "speaker_03",
  "tts_engine": "cosyvoice3",
  "context_id": "ctx_ship_0001_0001",
  "snr_db": 10,
  "speed_factor": 1.08,
  "pinyin_distance": 0.22,
  "character_distance": 0.5,
  "is_common_word": true,
  "suspected_tts_artifact": false
}
```

---

## 7. 上下文生成方案

## 7.1 每个术语的句型覆盖

每个术语至少覆盖以下上下文：

1. 句首；
2. 句中；
3. 句尾；
4. 陈述句；
5. 疑问句；
6. 否定句；
7. 指令句；
8. 定义句；
9. 并列术语；
10. 与相似术语共现；
11. 口语停顿；
12. 自我修正；
13. 省略表达；
14. 带数字和单位；
15. 中英文混合；
16. 会议口语；
17. 技术汇报；
18. 问答对话。

示例：

```text
外板纵骨的间距需要重新计算。
我们先检查一下外板纵骨。
这个位置是不是外板纵骨？
这不是外板纵骨的问题。
外板纵骨和舷侧纵桁都需要调整。
那个……外板纵骨，好像还差一根。
外板纵骨间距调整到六百毫米。
```

## 7.2 LLM 输出约束

LLM 必须输出结构化 JSON，不得自由增加未定义字段。

要求：

- 术语必须原样出现；
- 明确返回字符位置；
- 不能把标准术语改写成别名；
- 单句长度优先控制在 8～35 个汉字；
- 以会议口语为主；
- 同一批次避免模板高度重复；
- 对缩写术语同时生成不同合法 spoken form；
- 每个术语生成 12～30 个上下文。

## 7.3 上下文去重

使用三层去重：

1. 完全文本去重；
2. 字符 n-gram 相似度；
3. sentence embedding 相似度。

高于阈值的相似句只保留一条。

---

## 8. TTS 生成方案

## 8.1 音色维度

首轮覆盖：

- 男声、女声；
- 青年、中年；
- 低沉、明亮；
- 普通话；
- 轻微区域口音；
- 快速会议发言；
- 疲劳或低能量说话；
- 较远距离说话。

建议至少：

```text
8～12 个主音色
```

## 8.2 语速维度

建议采样：

```text
0.85～1.18
```

以连续随机值为主，不要只固定三个档位。

分布建议：

- 60%：0.95～1.08
- 20%：0.85～0.95
- 20%：1.08～1.18

## 8.3 情绪与表达方式

以中性为主：

- neutral：70%
- serious：10%
- tired：8%
- excited：5%
- hesitant：7%

避免极端戏剧化风格占比过高。

## 8.4 发音控制

对于以下内容必须显式指定 spoken form：

- 英文缩写；
- 中英文混合模型名；
- 数字型号；
- 生僻人名；
- 行业内部简称；
- 多音字术语。

每个 spoken form 独立生成样本，不要混在同一字段中。

---

## 9. 音频增强方案

建议使用 `audiomentations` 或等效工具。

## 9.1 增强类型

### 背景噪声

- 会议室多人低语；
- 空调噪声；
- 键盘声；
- 风扇声；
- 走廊声；
- 交通背景；
- 远端扬声器串音；
- 音乐低强度背景。

### SNR

建议：

- clean：30%
- 20 dB：20%
- 15 dB：20%
- 10 dB：20%
- 5 dB：10%

### 混响

- 无混响；
- 小会议室；
- 大会议室；
- 远场；
- 高反射房间。

### 设备与编码

- 原始 PCM；
- Opus；
- AAC；
- MP3；
- 低通；
- 重采样；
- 轻微削波；
- 麦克风增益变化。

## 9.2 增强原则

- clean 样本必须保留；
- 每条 clean 音频生成 0～2 个增强版本；
- 不允许所有增强同时达到极端值；
- 必须记录随机种子；
- 必须保存完整增强参数；
- 如果增强后不可懂，应标记为 invalid，而不是继续进入别名库。

---

## 10. TTS 质量控制

TTS 质量错误是整个方案最大的污染源之一。

## 10.1 自动检查

每条音频至少检查：

- 文件可读取；
- 采样率正确；
- 时长合理；
- 非静音；
- 无明显削波；
- RMS 在合理范围；
- spoken text 未被 TTS 截断；
- 音频时长与文本长度比例合理。

## 10.2 发音有效性检查

分层策略：

### 第一层：规则检查

- TTS 输入中显式使用正确 spoken form；
- 缩写和数字型号使用预定义发音。

### 第二层：交叉 ASR

使用第二个独立 ASR 对 clean 音频解码。

目的不是要求完全正确，而是过滤：

- 术语完全未发出；
- 句子严重缺失；
- TTS 明显读成其他词。

### 第三层：Forced Alignment

对高价值或高风险术语使用 forced alignment，检查目标术语是否能在预期时间区域对齐。

### 第四层：人工抽样

每个：

- TTS 引擎；
- 音色；
- spoken form；
- 术语类别；

至少抽样若干条人工听审。

## 10.3 TTS 无效标签

```text
tts_invalid_missing_term
tts_invalid_wrong_pronunciation
tts_invalid_truncated
tts_invalid_unintelligible
tts_invalid_audio_artifact
```

所有 `tts_invalid_*` 样本不得进入正式混淆词统计。

---

## 11. Qwen3-ASR 解码

## 11.1 必须固定配置版本

每次运行记录：

- 模型版本；
- 权重 hash；
- 推理代码 commit；
- decode 参数；
- chunk 参数；
- VAD 参数；
- 文本标准化版本；
- 运行设备；
- 随机种子。

## 11.2 流式结果保存

不能只保留 final。

需要记录：

- 每次 partial 文本；
- partial 时间戳；
- 术语首次出现时间；
- 错误术语首次出现时间；
- final 稳定时间；
- partial 修订次数；
- partial 到 final 的变化。

## 11.3 解码输出分类

```text
offline_final
streaming_partial
streaming_final
```

三个阶段独立统计。

---

## 12. 对齐与错误片段抽取

## 12.1 对齐层级

建议组合：

1. 中文字符级 Levenshtein 对齐；
2. 英文 token 级对齐；
3. 拼音或音素级对齐；
4. 目标术语局部窗口对齐。

## 12.2 抽取目标

输入：

```text
标准：我们先调整外板纵骨的布置方案
识别：我们先调整外版中国的布置方案
```

输出：

```json
{
  "reference_span": "外板纵骨",
  "hypothesis_span": "外版中国",
  "left_context": "我们先调整",
  "right_context": "的布置方案"
}
```

## 12.3 避免整句作为别名

错误抽取必须尽量局部化。

错误做法：

```text
canonical: 外板纵骨
alias: 我们先调整外版中国的布置方案
```

正确做法：

```text
canonical: 外板纵骨
alias: 外版中国
```

---

## 13. 错误 Tag 体系

## 13.1 ASR 错误类型

```text
exact
format_only
phonetic_substitution
common_word_substitution
partial_term
deletion
insertion
boundary_split
boundary_merge
acronym_expansion
acronym_reduction
code_switch_error
number_error
unit_error
itn_error
semantic_hallucination
streaming_instability
tts_invalid
unknown
```

## 13.2 风险类型

```text
safe_alias
context_required
hard_negative
rare_unstable
speaker_specific
tts_artifact
noise_only_error
streaming_only_error
global_replace_forbidden
```

## 13.3 来源维度

```text
clean
noise
reverb
speed
codec
accent
speaker
context
tts_engine
```

---

## 14. LLM 标注方案

## 14.1 LLM 输入

LLM 只能看到：

- 标准术语；
- 标准句子；
- ASR 实际输出；
- 已对齐错误片段；
- 拼音距离；
- 字符距离；
- 音频元数据；
- 可选上下文关键词。

## 14.2 LLM 输出

```json
{
  "error_type": "phonetic_substitution",
  "contains_common_word": true,
  "common_word_spans": ["中国"],
  "replacement_policy": "context_required",
  "suspected_tts_artifact": false,
  "requires_human_review": true,
  "reason_code": "COMMON_WORD_COLLISION"
}
```

## 14.3 禁止行为

LLM 不得：

- 新造 ASR 未输出过的别名；
- 自由改写整句；
- 删除低频但真实的错误；
- 仅凭语义决定是否纠正；
- 隐藏不确定性。

## 14.4 标注一致性

建议：

- 同一批数据使用固定 prompt；
- 固定模型版本；
- 固定 temperature 为低值；
- 对高风险样本重复标注两次；
- 使用规则和 LLM 交叉校验；
- 人工审核高价值样本。

---

## 15. 四类最终产物

## 15.1 原始混淆库存

保存所有实际错误：

```json
{
  "canonical": "外板纵骨",
  "hypothesis": "外版中国",
  "count": 18,
  "speaker_count": 6,
  "context_count": 9,
  "tts_engine_count": 2
}
```

## 15.2 安全别名表

仅包含可较安全直接归一化的别名：

```json
{
  "alias": "外版纵骨",
  "canonical": "外板纵骨",
  "replacement_mode": "direct",
  "confidence": 0.96
}
```

## 15.3 上下文规则表

```json
{
  "alias": "中国",
  "canonical": "纵骨",
  "replacement_mode": "contextual",
  "required_context": [
    "外板",
    "间距",
    "布置",
    "船体"
  ],
  "forbidden_context": [
    "国家",
    "地区",
    "发展"
  ]
}
```

## 15.4 Hard Negative 表

```json
{
  "phrase": "中国",
  "confusable_term": "纵骨",
  "text": "中国船舶工业的发展速度很快。",
  "expected_action": "keep_original"
}
```

---

## 16. 候选聚合与排序

## 16.1 聚合统计

每个混淆候选统计：

- 总出现次数；
- clean 出现次数；
- 不同说话人数；
- 不同上下文数；
- 不同 TTS 引擎数；
- 不同 spoken form 数；
- 不同增强条件数；
- partial 出现次数；
- final 出现次数；
- 真实会议日志出现次数；
- common word 频率；
- hard negative 命中率。

## 16.2 候选分数

初始可采用：

```text
candidate_score =
    0.20 × frequency_score
  + 0.15 × speaker_diversity
  + 0.15 × context_diversity
  + 0.15 × tts_engine_diversity
  + 0.10 × clean_reproducibility
  + 0.10 × phonetic_similarity
  + 0.10 × production_log_support
  - 0.15 × common_word_risk
  - 0.15 × hard_negative_false_replace_rate
  - 0.10 × tts_artifact_risk
```

首版权重可以通过经验设置，后续根据人工审核数据调整。

## 16.3 最低候选条件

进入人工审核前，建议至少满足：

- 跨 2 个音色出现；
- 跨 2 个上下文出现；
- 或在真实会议中出现；
- 不是单一 TTS 独有；
- 不是明显 TTS 无效发音；
- 局部拼音距离在合理范围。

---

## 17. 评估指标

## 17.1 混淆库存质量

- `Unique Confusions per Term`
- `Cross-Speaker Reproducibility`
- `Cross-Context Reproducibility`
- `Cross-TTS Reproducibility`
- `Clean Condition Reproducibility`
- `Real-Log Coverage`

## 17.2 别名表质量

- Alias Precision
- Alias Recall
- Direct Replacement Precision
- Contextual Replacement Precision
- False Replacement Rate
- Hard Negative Accuracy

## 17.3 ASR 术语指标

- Term Recall
- Term Precision
- Term Error Rate
- Keyword Error Rate
- Partial Term Recall
- Final Term Recall
- Streaming Revision Count
- Term Commit Latency

## 17.4 对下游意图识别的影响

- 意图准确率；
- 术语相关意图召回率；
- 误触发率；
- 首次意图命中延迟；
- 最终意图确认延迟；
- 纠错前后差值。

---

## 18. 真实会议数据验证

纯合成数据不能直接证明线上有效。

需要准备一小批真实会议术语测试集：

- 50～200 段真实语音；
- 覆盖主要术语；
- 覆盖不同说话人；
- 覆盖会议室噪声；
- 人工标注正确文本；
- 标注术语时间范围；
- 保存 Qwen3-ASR 原始输出。

验证：

1. 合成数据中的高频混淆是否出现在真实会议；
2. 真实会议中的混淆是否能被合成 Pipeline 覆盖；
3. safe alias 是否产生误替换；
4. context-required 规则是否有效；
5. hard negative 是否覆盖真实普通词场景。

---

## 19. 实施阶段

## 阶段 0：环境和样本准备

### 任务

- 确认 Qwen3-ASR 推理入口；
- 固定线上解码配置；
- 部署 CosyVoice 3；
- 选择第二路 TTS；
- 准备 MUSAN 或自有会议噪声；
- 准备 RIR；
- 选取 30～50 个术语；
- 定义术语 JSON Schema；
- 准备 50～200 段真实测试语音。

### 输出

- `terms.jsonl`
- `pipeline.yaml`
- 推理环境说明
- 真实测试集 manifest

### 验收

- 任意一个术语可通过命令完整生成 clean 音频；
- Qwen3-ASR 可对该音频完成 offline 和 streaming 解码；
- 所有配置可复现。

---

## 阶段 1：上下文与 TTS 生成

### 任务

- 开发 LLM 上下文生成脚本；
- 实现上下文去重；
- 接入两路 TTS；
- 实现 spoken form；
- 生成 clean 音频；
- 记录完整 manifest。

### 输出

- `generated_contexts.jsonl`
- `tts_manifest.jsonl`
- clean 音频目录

### 验收

- 每个术语至少 12 个有效上下文；
- 至少 8 个音色；
- 至少 2 个 TTS 来源；
- 人工抽检 TTS 发音有效率达到 95% 以上。

---

## 阶段 2：增强与 ASR 解码

### 任务

- 实现音频增强随机采样；
- 生成增强音频；
- 运行 Qwen3-ASR offline；
- 运行 Qwen3-ASR streaming；
- 保存 partial 和 final；
- 记录解码延迟。

### 输出

- `augmentation_manifest.jsonl`
- `asr_manifest.jsonl`
- ASR 原始输出

### 验收

- 5,000～12,000 条音频完成解码；
- 失败样本有明确错误记录；
- 95% 以上有效样本有完整输出；
- partial 和 final 可关联到同一 sample。

---

## 阶段 3：对齐与混淆抽取

### 任务

- 实现字符级对齐；
- 实现拼音级距离；
- 实现术语局部窗口抽取；
- 过滤 exact；
- 标记 TTS invalid；
- 聚合混淆候选。

### 输出

- `aligned_manifest.jsonl`
- `confusion_inventory.jsonl`
- 初步统计报告

### 验收

- 随机抽查 200 条对齐结果；
- 术语错误片段抽取准确率达到 90% 以上；
- 整句误抽取比例低于 3%。

---

## 阶段 4：LLM 标注与人工审核

### 任务

- 实现规则标签；
- 实现 LLM 标签；
- 识别常见词冲突；
- 生成 hard negative；
- 导出审核表；
- 人工审核 Top-K。

### 输出

- `llm_labeled_confusions.jsonl`
- `review_candidates.csv`
- `hard_negatives.jsonl`
- 审核记录

### 验收

- 每个高频候选都有风险类型；
- safe alias 经过人工确认；
- 所有 common word 冲突进入上下文规则或 hard negative；
- LLM 不新增 ASR 未观测别名。

---

## 阶段 5：规则生成与离线评测

### 任务

- 生成 safe alias；
- 生成 context-required 规则；
- 生成 hard negative 测试集；
- 在真实语音上评估；
- 评估意图识别变化。

### 输出

- `safe_aliases.json`
- `contextual_rules.json`
- `hard_negative_eval.jsonl`
- `evaluation_report.md`

### 验收建议

- safe alias precision ≥ 98%；
- contextual correction precision ≥ 95%；
- hard negative 保留准确率 ≥ 98%；
- 术语召回率有明显提升；
- 意图误触发率不高于基线；
- 解码后处理增加延迟可控。

---

## 20. CLI 设计

建议提供统一入口：

```bash
python -m src.cli generate-contexts \
  --terms data/terms/terms.jsonl \
  --output data/contexts/generated_contexts.jsonl

python -m src.cli generate-tts \
  --manifest data/contexts/generated_contexts.jsonl \
  --config config/tts.yaml

python -m src.cli augment \
  --manifest data/manifests/tts_manifest.jsonl \
  --config config/augmentation.yaml

python -m src.cli decode \
  --manifest data/manifests/augmentation_manifest.jsonl \
  --config config/asr.yaml \
  --mode both

python -m src.cli mine-confusions \
  --manifest data/manifests/asr_manifest.jsonl

python -m src.cli label-confusions \
  --input data/confusion_inventory/confusions.jsonl

python -m src.cli build-alias-tables \
  --reviewed data/reviewed/reviewed_confusions.jsonl
```

---

## 21. 配置示例

```yaml
project:
  name: qwen3_asr_term_confusion_mining
  seed: 20260716

terms:
  max_terms: 50
  contexts_per_term: 20
  samples_per_term: 200

tts:
  engines:
    - name: cosyvoice3
      weight: 0.7
    - name: secondary_tts
      weight: 0.3
  speakers_per_term: 8
  speed_range: [0.85, 1.18]

augmentation:
  clean_probability: 0.3
  snr_db_choices: [20, 15, 10, 5]
  reverb_probability: 0.35
  codec_probability: 0.2
  max_augmented_versions_per_clean: 2

asr:
  model: Qwen3-ASR-1.7B
  modes:
    - offline
    - streaming
  save_partials: true
  save_latency: true

mining:
  min_speaker_count: 2
  min_context_count: 2
  min_tts_engine_count: 1
  use_pinyin_alignment: true

llm_labeling:
  enabled: true
  temperature: 0.1
  require_json: true
  forbid_new_aliases: true
```

---

## 22. 人工审核表字段

建议导出 CSV 或 Excel：

| 字段 | 说明 |
|---|---|
| canonical | 标准术语 |
| hypothesis | ASR 混淆片段 |
| total_count | 总次数 |
| clean_count | 干净音频次数 |
| speaker_count | 音色覆盖 |
| context_count | 上下文覆盖 |
| tts_engine_count | TTS 覆盖 |
| real_log_count | 真实日志次数 |
| pinyin_distance | 拼音距离 |
| common_word | 是否常见词 |
| llm_error_type | LLM 错误类型 |
| proposed_policy | 建议策略 |
| human_decision | 人工决定 |
| reviewer_note | 备注 |

人工决定枚举：

```text
safe_alias
context_required
hard_negative
ignore
tts_artifact
need_more_samples
```

---

## 23. 风险与应对

## 23.1 TTS 发音污染

### 风险

TTS 错读术语，导致错误被误认为 ASR 混淆。

### 应对

- 显式 spoken form；
- 双 TTS；
- 交叉 ASR；
- forced alignment；
- 人工抽检；
- 单一 TTS 独有错误降权。

## 23.2 合成音频与真实会议分布差异

### 风险

合成音频错误在真实会议中不出现。

### 应对

- 使用真实会议参考音色；
- 使用真实会议噪声；
- 加入真实设备和编码模拟；
- 用真实会议日志作为最终验证；
- 将真实日志支持度纳入候选分数。

## 23.3 常见词误替换

### 风险

例如“纵骨”错误映射到“中国”，全局替换会破坏正常句子。

### 应对

- 四表分离；
- hard negative；
- 上下文触发；
- 禁止默认全局替换；
- 评估 false replacement rate。

## 23.4 数据组合爆炸

### 风险

完整组合导致成本过高。

### 应对

- 分层随机采样；
- 自适应追加样本；
- 稳定术语提前停止；
- 高风险术语增加采样；
- 缓存 clean 音频和 ASR 结果。

## 23.5 LLM 标签不稳定

### 风险

LLM 对相同候选给出不同结论。

### 应对

- 低 temperature；
- 固定 prompt 和模型版本；
- 规则优先；
- 双次标注；
- 人工审核高频候选；
- 禁止新增别名。

## 23.6 流式错误与 final 错误混淆

### 风险

仅在 partial 出现的错误被当作最终别名。

### 应对

- partial 和 final 分开建表；
- 增加 `streaming_only_error`；
- 依据下游是否消费 partial 决定处理；
- 单独评估术语 commit latency。

---

## 24. 成功标准

MVP 成功需要满足：

1. 能端到端自动生成术语测试音频；
2. 能稳定调用 Qwen3-ASR 离线和流式解码；
3. 能抽取目标术语对应的局部错误片段；
4. 能区分 safe alias、context-required、hard negative 和 TTS artifact；
5. 在真实会议语音中验证部分合成混淆具有实际覆盖；
6. 使用别名或规则后，术语召回率提高；
7. 常见词误替换率保持在可接受范围；
8. Pipeline 可重复运行，所有样本可追溯。

---

## 25. 第一轮实验建议

### 术语

选择 40 个：

- 10 个中文行业术语；
- 10 个英文缩写；
- 10 个中英文混合术语；
- 5 个人名或项目名；
- 5 个数字型号。

### 样本

每个术语：

- 16 个上下文；
- 8 个音色；
- 2 个 TTS；
- 随机生成约 200 条；
- 30% clean；
- 70% 不同程度增强。

总量约：

```text
40 × 200 = 8,000 条
```

### 第一轮重点观察

- 每个术语产生多少 unique confusion；
- confusion 是否跨音色复现；
- confusion 是否跨 TTS 复现；
- clean 与 noise 下错误差异；
- partial 与 final 差异；
- common word collision 比例；
- 真实会议日志覆盖率；
- safe alias 可用比例。

---

## 26. 后续扩展方向

MVP 验证成功后，可继续：

1. 将混淆词库存用于 ASR 后处理；
2. 构建轻量分类器判断是否替换；
3. 使用 SpellMapper 类方法做候选检索和纠错；
4. 将 hard negative 用于训练；
5. 根据当前会议主题动态加载术语；
6. 利用 partial 序列预测术语最终稳定结果；
7. 将真实线上错误回流到混淆词库存；
8. 评估 Qwen3-ASR 领域微调；
9. 研究原生 contextual biasing；
10. 建立按项目、会议、参会人动态更新的术语服务。

---

## 27. 最小可执行顺序

```text
1. 定义 40 个术语
2. 为每个术语生成 16 个上下文
3. 接入 CosyVoice 3
4. 先生成 500 条 clean 音频
5. 跑 Qwen3-ASR offline
6. 实现字符级和拼音级对齐
7. 检查能否稳定抽取混淆词
8. 再接入第二路 TTS
9. 再加入噪声和混响
10. 再跑 streaming partial
11. 加入 LLM 标签和 hard negative
12. 最后在真实会议数据上验证
```

不建议一开始就同时搭建全部组件。首先用 5～10 个术语和 500 条音频验证：

```text
TTS → Qwen3-ASR → 对齐 → 混淆抽取
```

该核心链路可靠后，再扩展到完整 MVP。
