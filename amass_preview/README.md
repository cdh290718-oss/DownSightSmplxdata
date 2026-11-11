# ACCAD 与 CNRS 双目合成预览

本目录整理 6 组使用 ACCAD 或 CNRS 动作数据生成的下视双目预览和三维标签。用户确认持有公开发布许可；后续使用或再分发者仍须核实数据集、SMPL-X、环境贴图等各自适用的许可。

| 样例 | 来源动作文件 | 帧号（从 0 开始） | 说明 |
|---|---|---:|---|
| [accad_walk](samples/accad_walk) | `ACCAD/s007/QkWalk1_stageii.npz` | 153 | 行走 |
| [accad_run](samples/accad_run) | `ACCAD/Female1Running_c3d/C5_-_walk_to_run_stageii.npz` | 325 | 跑步；人体多数在画面外 |
| [accad_kick](samples/accad_kick) | `ACCAD/Male2MartialArtsKicks_c3d/G18-__push_kick_right_stageii.npz` | 173 | 踢腿；手部有遮挡 |
| [accad_crouch](samples/accad_crouch) | `ACCAD/Male1General_c3d/General_A10_-__Lie_Down_to_Crouch_stageii.npz` | 636 | 屈腿；左右手、腿部可见，部分遮挡 |
| [accad_swing](samples/accad_swing) | `ACCAD/Female1General_c3d/A3_-_Swing_stageii.npz` | 512 | 摆动；肢体部分遮挡 |
| [cnrs_01](samples/cnrs_01) | `CNRS/288/-01_L_1_stageii.npz` | 370 | CNRS 示例；双手及部分双腿可见 |

图像分辨率 1280×720；左右均使用仓库 `calibration/stereoParams_20260604_fixed_native.json` 旧 Brown–Conrady 标定，基线约 0.15447 m，**未使用新鱼眼标定**。SMPL-X 模型使用源性别与 16 个 betas、`use_pca=False`、`flat_hand_mean=True`；将全局方向从 Z-up 转为场景 Y-up，并将 pelvis 重新定位至参考人体附近。相机刚性随 SMPL-X 头部变化，安装位置继承旧场景，未经真实设备测量。环境为 HDR 贴图而非有深度的地面，人体使用简化材质。

每个样例包括 `left.png` / `right.png`、`smplx_pose.npz`（参数、局部顶点和关节、头部矩阵）、`geometry.npz`（网格、相机与世界变换）、`left_vertices.npz` / `right_vertices.npz`（相机坐标、二维投影和入画标记）、`rig_reference.npz` 与 `validation.json`。长度单位为米，相机坐标为 OpenCV 约定；投影采用假设的 MATLAB 1 基到零基转换。入画标记不判断遮挡；环境贴图和人物不保证接触关系。各帧为动作拟合结果驱动的合成真值，并非真实相机实拍真值或所拍人物精确重建。

筛选在 18 段 ACCAD、16 段 CNRS 序列中抽样，基于关节点入画、姿态变化和低分辨率表面遮挡近似评分；不同动作并未穷尽。静态帧的筛选不能保证动作连续时的稳定性。CNRS 的 3 段抽查序列手指姿态基本不变，不可推断全部 CNRS 动作均无手指运动。原始动作压缩包、模型文件、场景和 HDR 不在此目录。

## 校验与重新筛选

在仓库根目录运行 `python amass_preview/verify_samples.py`，校验六组图像尺寸、双目位姿、顶点和关节投影；它是内部几何一致性检查，不验证现实标定或遮挡。

如已自行取得动作、SMPL-X 模型及依赖，可使用工作区内已有的头部 rig 和旧标定重跑筛选：

```powershell
$env:DOWNSIGHT_AMASS_DATASET = 'CNRS'  # 或 ACCAD
$env:DOWNSIGHT_AMASS_ARCHIVE = 'D:\path\CNRS.tar.bz2'
$env:DOWNSIGHT_SMPLX_MODEL_DIR = 'D:\path\models'  # 包含 smplx 模型子目录
python amass_preview/select_visible.py
```

输出默认写入 `generated_output/amass_preview/`。该脚本需要 NumPy、SciPy、PyTorch、SMPL-X、OpenCV 和 Blender 渲染环境；筛选本身不运行 Blender。使用仓库内的 `render_stereo_pano.py` 和 `remap_stereo.py` 渲染候选前，还需要自行取得兼容 Blender 场景及 HDR，参见 [MCAP 示例说明](../mcap_conversion/README.md)。
