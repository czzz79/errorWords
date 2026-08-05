# GT3 英文/缩写 CMU 音素实验：复用 clean TTS，只测试五种单项音频扰动。
# 输出为 359 x 5 = 1,795 条 ASR 输入；不会重新生成 CosyVoice 音频。
python run_pipeline.py --config experiments/configs/gt3-english-cmu-cosy-single-augmentations.json
