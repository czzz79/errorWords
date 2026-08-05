# CMU 音素英文预处理：canonical -> CMUdict ARPAbet -> 单步 target-blind 音素扰动。
# 只生成 pronunciation manifest，不启动 CosyVoice、音频扰动或 ASR。
python run_pipeline.py --config experiments/configs/gt3-english-cmu-preprocess.json
