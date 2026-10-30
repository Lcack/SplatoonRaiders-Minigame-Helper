# 本地参考实况

2026-10-30 已下载用户指定的 UTAGaming 参考实况，保存视频与原声并合并为 MP4。之后的关键帧核查可使用本地文件。

项目内路径：`artifacts/reference_videos/utagaming-l3dMQV1sar8/reference.mp4`。

本地核查结果：

- 1920×1080，60 帧/秒，H.264 视频与双声道 AAC 音频。
- 视频流时长约 21 分 42.3 秒；含音频的容器时长约 21 分 42.4 秒。
- 文件 448,356,963 字节，约 427.6 MiB。
- 已抽样解码开头、中间和结尾，已检查容器中的音视频流；没有宣称逐帧解码完整视频。
- SHA-256：`58bbc11ae82d4bae106439a1374f6d0248b4f2de889167626bf7ea49a7cff071`。

同目录的 `source.json` 保存来源、作者、日期、章节、视频参数和文件校验值；`reference.description` 保存简介，`reference.info.json` 保存下载工具的原始元数据。

## 章节索引

| 起止时间 | 内容 |
|---|---|
| 00:00–05:25 | MINING |
| 05:25–11:03 | HOPPING |
| 11:03–16:03 | JUGGLING |
| 16:03–20:33 | LANDING |
| 20:33–结尾 | Credits |

时间来自来源视频章节元数据。[1]

## 离线使用

在项目目录运行：

```powershell
python run.py gui
```

选择“视频回放”，选择对应小游戏并载入上述 MP4，即可播放、暂停、单帧检查和按秒跳转。实机 profile 尚未校准，播放视频不代表已经能识别视频中的所有对象。

抽取 MINING 开头的样本：

```powershell
python run.py extract artifacts/reference_videos/utagaming-l3dMQV1sar8/reference.mp4 --start 5 --end 15 --step 0.25 --output artifacts/mining_samples
```

下载工具为 `tools/download_reference.py`，调用当前 conda Python，下载依赖隔离放在 `artifacts/tooling`。这些工具不作为游戏运行器依赖，也未安装进 base 环境。`artifacts` 已被 `.gitignore` 排除，不会默认加入代码版本管理。

## 本轮新增的离线观察

MINING 的 00:39.50 帧仍能看到前方红卵，00:39.75–00:40.25 出现 `+2.0` 提示、角色变为强化外观。00:39.50 的倒计时显示 29.5，00:39.75 显示 31.3；计时仍在自然流逝，因此不能直接用两帧数值的差替代事件加时。`+2.0` 的游戏提示与攻略大百科的加时表一致，支持将吃卵本身 +2 秒从单源待验证项升级为已采用规则。[1][2]

相关本地证据：同目录的 `mining-egg-37-41.jpg` 为每 0.25 秒的观察拼图，`mining-egg-evidence.json` 保存单帧索引及说明。`mining-overview-1.jpg`、`mining-overview-2.jpg` 是 MINING 前 320 秒每 5 秒抽样的概览。

## 来源

[1] [UTAGaming 原始实况](https://www.youtube.com/watch?v=l3dMQV1sar8)。本地文件和章节的来源。

[2] [攻略大百科 MINING 加时表](https://gamepedia.jp/splatoon-raiders/archives/3430)。红卵 +2.0 秒的二级资料，用本地游戏画面交叉验证。

声明分类与置信度：本地文件参数、校验值、抽样解码和直接画面观察属于客观事实，置信度高（≥80%）；将游戏加时提示与资料合并成规则属于逻辑推断，置信度高（≥80%）。尚未量化完整游戏的运动参数。
