# 头部安装变换

`rig_reference.npz` 记录旧参考姿态下 head 到左右 Blender 相机的固定变换；`reference_pose.npz` 是参考姿态及 head 位姿，`head_yaw25.npz` 是数值测试姿态。`test_head_rig.py` 检查参考姿态无跳变、头部旋转后相对变换不变，以及整体刚体运动等变。

渲染器按 `head_to_world = object_to_world @ head_to_model` 和 `camera_to_world = head_to_world @ head_to_camera` 更新每一帧。左右标定与安装变换延续旧样例，尚未使用 2026-11-08 新标定或实测头部安装位姿。参考姿态的双目中点相对 head 的世界坐标差约为 `(0.0111, 0.1940, 0.0818)` 米，旧左相机向下约 80°；这些只是旧场景数据。
