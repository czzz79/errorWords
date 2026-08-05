# 仅运行 GT 自动补全声母/韵母规则新增的文本载体；复用上一轮已有音频结果，不重复合成 canonical 或静态规则样本。
python run_pipeline.py --config experiments/configs/gt3-zh-fuzzy-pinyin-derived-components-delta-matrix.json
