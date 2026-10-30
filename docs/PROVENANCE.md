# 模块来源

项目最初从用户指定的 `E:\Splatoon\tableturf\AutoTableTurf-rebuild` 读取并复制以下模块，包名从 `tableturf_rebuild` 改为 `raiders_helper`：

- `control/__init__.py`、`backend.py`、`buttons.py`、`dummy.py`、`easycon.py`、`debug_ui.py`。
- `capture/models.py`、`opencv_source.py`、`video_source.py`、`recorder.py`。
- `tests/test_control.py`。

`control/debug_ui.py` 的窗口标题改为 Raiders。`capture/worker.py` 根据原项目的最新帧线程结构重写：停止时由采集线程自行关闭设备，避免读取未退出时跨线程 release；重新启动会清空旧帧和错误。

本项目新增的识别、策略、调试和分析模块包括 `vision.py`、`arcade_vision.py`、`fast_tracking.py`、`hopping.py`、`juggling.py`、`landing.py`、`mining_planner.py`、`analysis_worker.py`、`video_analysis.py`，以及 guard、runtime、GUI、CLI 等。三项快速策略没有从旧打牌项目复制游戏规则或实机运动参数。

## 参考视频与派生视觉资料

用户指定的参考实况是 UTAGaming 的 [四项小游戏视频](https://www.youtube.com/watch?v=l3dMQV1sar8)，本地文件、来源与校验信息见 [LOCAL_VIDEO.md](LOCAL_VIDEO.md)。本项目未把其中操作者的成绩当作 AI 实际取得的成绩。

- `assets/arcade/source.json` 记录 HUD 锚点来源；HOPPING 卵和金色目标模板的时间、裁剪与来源保存在 `assets/arcade/hopping-provenance.json`。对应研究片段位于 `artifacts/hopping_egg_reference`，卵与强化外观分别保存，未仅凭普通鱼体的红色断言为卵。
- `maps/landing/index.json` 记录这局十个布局的原始样本。由其生成的地形掩膜、姿态参考、匹配资料及几何路点位于 `assets/landing_atlas`，来源信息在该目录的 `index.json`；这些资料仅覆盖这一局观察到的布局。
- `tools/build_landing_atlas.py` 可从已保存样本重建派生资料并做同局稀疏时刻检查。来源帧与同局其他时刻仍共享视频、画面风格及布局，不能将这种检查称为独立视频泛化评估。

这些资料用于参考录播的颜色、形状和模板分析；不是训练得到的通用检测模型，也不是已校准的采集卡/手柄控制模型。动作建议、匹配分数及 ready/advice 帧数的含义和最终计时范围见 [VIDEO_ANALYSIS.md](VIDEO_ANALYSIS.md)。

原项目的 `pyproject.toml` 声明 `GPL-3.0-or-later`，本项目保留该声明，并复制原 `LICENSE` 文件。原始来源与现存作者信息由原项目保留，本说明不新增未知作者归属。

协议约定来自被移植的 `easycon.py` 与 `CONTROLLER_DEBUG.md`：115200/8N1、握手及停止固件脚本命令、7-bit 报告打包、报告不等待逐包 ACK。当前测试只验证这些代码的行为，没有验证新主机上的硬件兼容性。

声明分类：以上为本地文件核查得到的客观事实，置信度高（≥80%）。
