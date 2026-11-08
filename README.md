# DownSight SMPL-X 双目合成真值：实验样例

本仓库提供一对下视双目渲染图，以及与渲染人体对应的 SMPL-X 参数、网格、关节和相机数据。当前只有一个姿态、一对图像，供 EgoSMPLX 推理与误差评测的流程联调。

## 文件

`samples/000001/left.png`、`right.png` 分别是 cam3 和 cam4 的 1280×720 输入图。`smplx_reconstruction.npz` 是用于生成网格的 SMPL-X 参数及局部顶点/关节；`geometry.npz` 是实际渲染网格的世界坐标、网格拓扑、127 个关节、对象矩阵和两目位姿。`left_vertices.npz`、`right_vertices.npz` 保存每目 3D/2D 顶点和 127 个关节。`left_egosmplx65.npz`、`right_egosmplx65.npz` 使用 EgoSMPLX 的 25 个身体点加 40 个手部点顺序，每个文件包含名称、SMPL-X 索引、世界坐标、相机坐标、像素坐标和入画标记。名称分别见 `joint_names_smplx127.json` 和 `joint_names_egosmplx65.json`。

模型文件、人体贴图与本机输入图未上传。网格及参数属于此合成样例；使用 SMPL-X 模型进行重建须自行取得相应模型及使用授权。

## 坐标和评测

- 3D 长度单位为米；相机坐标为 OpenCV 约定：X 向右、Y 向下、Z 向前。世界坐标为 Blender 场景坐标。`world_to_camera_cv` 从世界坐标变换到该相机坐标。
- 2D 像素坐标以图像左上像素中心为 `(0,0)`，X 向右、Y 向下。假设原 MATLAB 标定主点采用 1 基坐标，转为零基时减 1；此假设仍需与原始 MATLAB 投影结果核实。负深度点的 2D 坐标为 NaN。
- `in_frame_no_occlusion_test` 仅表示点位于相机前方且投影在画幅内，**不是可见性或遮挡标签**。
- `smplx_reconstruction.npz` 中的 SMPL-X 参数为局部人体坐标系。把局部顶点或关节变到世界坐标还需乘 `geometry.npz` 的 `object_matrix_world`；不能直接把局部姿态参数与相机坐标下的预测参数比较。建议先比较同名关节的相机坐标，再分别报告绝对误差和 pelvis 对齐误差。
- `python evaluate_joints.py --prediction PRED.npz --side left` 支持对比已有 `smplx_body_joints_3d_render_cam` 字段（EgoSMPLX 25 身体点顺序，米）或者 `joints_3d_camera_m`（65 点顺序，米），输出每点与平均误差。使用前确认预测坐标系和尺度；该工具不执行推理。

## 几何与限制

内参使用 `calibration/stereoParams_20260604_fixed_native.json` 中原始 Brown–Conrady 模型；其他两个文件为 EgoSMPLX 使用的多项式拟合，**不是相同投影模型**，其拟合误差记录在各自文件中。渲染先生成 4096×2048 等距柱状图，再对目标相机像素反向采样；插值和全景分辨率会产生采样误差。两个相机的中点位于旧 Blender 场景相机位置；双目相对位姿沿用标定，基线约 154.5 mm。渲染实际网格与 SMPL-X 前向计算的最大顶点差为 0 m。

`validation.json` 的像素逆映射残差验证公式的内部一致性；`marker_validation.json` 是三个中心附近标记在实际渲染中的投影误差，约 0.12–0.25 px。它们均不能证明全视场误差或真实设备标定的正确性。历史双目外参尚未针对当前设备与 session_hand6 验证；相机相对人体的安装位姿沿用旧场景、未经实测。HDR 背景没有真实地面深度，不保证人物接触关系。此样例是真实生成过程的合成真值，不能直接视为真实设备几何已经完全匹配，也不能将实拍误差全部归因为模型本身。

## 检查

安装 `numpy` 和 `opencv-python` 后运行 `python validate_sample.py`；该检查验证图片尺寸、数据对应、相机变换与投影计算。验证结果不替代真实标定板或全视场标记检验。
