# MCAP → SMPL-X 转换实验

输入为 Gen-HumanWholeBody 的 MCAP 单文件（SHA-256 `00f34e66eacf623d03564571608665a5f6893be102f42c492785697c376e95c1`）。这里保存第 0、15、30、450、900、1350 帧的追踪数据节选 `raw_*.npz`、转换后 `smplx_*.npz`、提取和验证报告；完整 MCAP、SMPL-X 模型、Blender 原始场景及 HDR 文件未包含。源数据为 [Gen-HumanWholeBody](https://huggingface.co/datasets/genrobot2025/Gen-HumanWholeBody)，使用、复制与公开时需遵守其数据访问条件和适用许可。

`render_0450_final/` 是第 450 帧的双目图、SMPL-X 参数、渲染网格、相机位姿和每目关节投影；`left_egosmplx65.npz` 与 `right_egosmplx65.npz` 为 EgoSMPLX 的 65 点顺序。入画标志未检测遮挡。内参和双目基线沿用仓库根目录旧的 `calibration/stereoParams_20260604_fixed_native.json`，不对应 `calibration/2026-11-08_lower_fisheye/` 新标定。该帧的姿态文件也复制在渲染目录，避免和以后重新拟合的结果混淆。

身体 22 点的顺序由空间结构推断为 SMPL-X 常见顺序；位置通过反向运动学拟合，身体追踪四元数没有使用。头部局部旋转设为零，无法认为复现原始头部方向。人体 betas 沿用静态样例，面部为零；源世界位置被重置至参考人体附近。左右手手腕朝向根据各自追踪点对齐估计；左手旋转使用 `F R F`，其中 `F=diag(-1,1,1)`，右手直接使用。该约定由样例点位对照支持，但并未得到数据集官方确认。

六帧身体 22 点平均拟合误差约 1.81–2.34 cm。左右手分别做刚体对齐后误差约 6–7 mm；放在身体共同坐标系后平均误差约 21–36 mm。前者不能用作手相对身体位置精度。`validation.json` 另检查模型重建、旋转矩阵和双目变换。这些验证表明保存参数和渲染数据一致，不代表准确重现源人物、真实设备安装几何或整套数据集。

## 重现

需要 Python 3.14（使用标准库 `compression.zstd`）、NumPy、SciPy、PyTorch、`smplx`、OpenCV，以及自行取得的 SMPL-X neutral 模型。运行以下 PowerShell 命令可在忽略的 `generated_output/` 下重新转换；不会覆盖仓库内的已发布样例：

```powershell
$env:DOWNSIGHT_MCAP_PATH = 'D:\path\input.mcap'
$env:DOWNSIGHT_SMPLX_MODEL_DIR = 'D:\path\models' # 包含 smplx\SMPLX_NEUTRAL.npz
python mcap_extract_motion.py
python convert_mcap_to_smplx.py
python validate_mcap_conversion.py
python test_head_rig.py
```

若要生成同类渲染，还需要可用的 Blender 场景、HDR 和 Blender 5.1。设定 `DOWNSIGHT_OUTPUT_DIR`、`DOWNSIGHT_POSE_PATH`、`DOWNSIGHT_HDR_PATH`，以后台模式打开兼容的旧场景并运行 `render_stereo_pano.py`，再执行 `python remap_stereo.py`。渲染脚本使用旧标定，且要求场景中的人体对象变换与保存的参考一致。HDR 可更换，但改变 HDR 会改变背景；重新生成不保证像素完全一致。
