# 中文模糊拼音规则来源

规则文件 `chinese-fuzzy-pinyin-rules.csv` 中的 `origin` 字段可追溯到以下来源。

- `ibus_libpinyin`：2026-08-03 在线核验 ibus-libpinyin 公开配置。其 `PYPConfig.cc` 和设置界面明确列出 `c/ch`、`z/zh`、`s/sh`、`l/n`、`f/h`、`l/r`、`g/k`、`an/ang`、`en/eng`、`in/ing` 模糊拼音开关。
  - https://github.com/libpinyin/ibus-libpinyin/blob/main/src/PYPConfig.cc
  - https://github.com/libpinyin/ibus-libpinyin/blob/main/setup/ibus-libpinyin-preferences.ui
- `manual_spec`：来自本项目的《中文发音混淆文本扰动生成说明（第一版）》；尚未在公开配置中逐条确认，保留为可配置扩展规则。
- `fcitx5`：2026-08-03 在线核验 Fcitx5 Chinese Addons 的 `FuzzyConfig`。其中可被带声调汉语拼音反解为中文载体的新增方向为 `u/ou`；`ue/ve` 与 `ng/gn` 是输入拼写兼容，不作为 TTS 文本扰动规则。
  - https://github.com/fcitx/fcitx5-chinese-addons/blob/master/im/pinyin/pinyin.h
- Microsoft Pinyin 公开说明确认提供 Fuzzy Pinyin 功能，但该页面未公开具体方向映射，因此不单独作为本 CSV 的映射依据。
  - https://support.microsoft.com/en-us/windows/hardware/input-devices/microsoft-simplified-chinese-ime

载体文本不依赖在线资源，仍只使用仓库本地的雾凇词典、高频词表和 jieba 词典。
