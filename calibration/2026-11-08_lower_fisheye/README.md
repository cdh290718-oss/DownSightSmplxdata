# 2026-11-08 下视双鱼眼标定

本目录保存重新调整镜头后、分别标定左右内参再估计双目外参的结果，适用于当时的焦距、对焦、相机相对姿态与 1280×720 采集模式。

- `stereo_fisheye_intrinsics_extrinsics.mat`：MATLAB 鱼眼内参对象 `leftIntrinsics`、`rightIntrinsics` 和双目外参 `stereoExtrinsics`；只包含参数，不包含照片或采集目录。
- `fisheye_intrinsics.json`：MATLAB Scaramuzza 鱼眼内参，**不能直接当作 OpenCV fisheye 畸变系数或 EgoSMPLX 相机多项式**。
- `stereo_extrinsics.json`：左到右双目相对位姿；文件内注明行向量和列向量约定，平移单位为毫米。
- [详细分析和验证](analysis.md)，[角点覆盖图](evidence/corner_coverage.png)。

标定板格长按 15 mm 假设，实际格长需要确认，基线和平移随实际格长同比变化。该套参数尚未转换或接入推理，也未用于生成仓库内已有的合成样例；样例仍对应旧标定。外围覆盖改善，但极边缘的独立验证还不充分。
