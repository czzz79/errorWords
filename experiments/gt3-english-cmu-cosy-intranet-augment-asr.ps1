# 内网续跑：下载 generatedaudios ZIP 并解压到 inputs/generatedaudios 后执行。
# 输入：357 条英文 CMU/CosyVoice clean WAV；输出：5 x 357 扰动音频、ASR JSONL 和报告。
python run_pipeline.py --config experiments/configs/gt3-english-cmu-cosy-intranet-augment-asr.json
