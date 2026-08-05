# 内网传输说明

本仓库只包含 ErrorWords 管线源码、实验配置、测试和资源；`outputs/` 下的
TTS、音频扰动、ASR 和报告结果均不会提交。

## 外部依赖

本地使用 CosyVoice 官方源码仓库：

```text
https://github.com/FunAudioLLM/CosyVoice.git
commit: 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
```

内网环境应将该仓库镜像到可访问地址，或在同一路径检出以上提交：

```text
third_party/CosyVoice
```

模型权重不在本仓库中。CosyVoice 所用权重为
`Fun-CosyVoice3-0.5B-2512`，应由内网模型仓库或服务单独提供。

## 音频传递

每次待交给内网后处理的 TTS 结果应作为独立、不可变的音频包，至少包含：

- WAV 文件；
- 对应的 `tts-manifest.jsonl`（或实验 manifest）；
- 一份 README，记录源实验、TTS 配置、音频数量与校验值。

不要把完整 `outputs/` 目录提交进源码仓库。小型包可直接放 Git LFS；较大的
包应使用内网制品库/对象存储，Git 只提交 manifest、校验值和下载位置。

## 内网续跑

将 `generatedaudios` 仓库 ZIP 解压到源码仓库下的：

```text
inputs/generatedaudios/
```

内网使用 `uv` 管理 Python 环境。首次运行会根据 `pyproject.toml` 安装本项目及其依赖；
也可以先执行 `uv sync`。然后修改内网 ASR 地址并运行：

```powershell
uv run python run_pipeline.py --config experiments/configs/gt3-english-cmu-cosy-intranet-augment-asr.json
```

该配置不启动或停止 ASR，只等待外部服务端口；它会生成 5 种单项音频扰动，
共 1,785 条输入，再通过 `/v1/audio/transcriptions` 执行 ASR。
