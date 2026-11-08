"""Validate sample geometry and saved projections; not a physical calibration test."""
from pathlib import Path
import json
import cv2
import numpy as np

root=Path(__file__).resolve().parent; s=root/'samples/000001'
g=np.load(s/'geometry.npz',allow_pickle=False); model=np.load(s/'smplx_reconstruction.npz',allow_pickle=False)
cal=json.loads((root/'calibration/stereoParams_20260604_fixed_native.json').read_text(encoding='utf8'))
index=json.loads((root/'joint_names_egosmplx65.json').read_text(encoding='utf8'))
idx=np.asarray(index['smplx127_indices']); names=index['names']; assert len(names)==len(idx)==65
M=g['object_matrix_world']; verts=model['vertices_local_m']@M[:3,:3].T+M[:3,3]; joints=model['joints_local_m']@M[:3,:3].T+M[:3,3]
vertex_error=float(np.max(np.abs(verts-g['vertices_world_m']))); joint_error=float(np.max(np.abs(joints-g['joints_world_m'])))
assert vertex_error<1e-6 and joint_error<1e-6
S=np.diag([1.,-1.,-1.,1.]); A=np.array(cal['poseCamera2']['A']); A[:3,3]/=1000
left=g['left_camera_to_world_blender']@S; right=g['right_camera_to_world_blender']@S
stereo_error=float(np.max(np.abs(np.linalg.inv(right)@left-A))); assert stereo_error<1e-6
results={}
for side,camera in [('left','camera1'),('right','camera2')]:
    image=cv2.imdecode(np.fromfile(s/(side+'.png'),dtype=np.uint8),cv2.IMREAD_COLOR); assert image.shape==(720,1280,3)
    d=np.load(s/(side+'_vertices.npz'),allow_pickle=False); ego=np.load(s/(side+'_egosmplx65.npz'),allow_pickle=False)
    K=np.array(cal[camera]['K']); K[:2,2]-=1; dist=np.array([*cal[camera]['radialDistortion'],*cal[camera]['tangentialDistortion'],0.])
    assert np.allclose(d['K_zero_based'],K) and np.allclose(d['distortion'],dist)
    xyz=joints@d['world_to_camera_cv'][:3,:3].T+d['world_to_camera_cv'][:3,3]
    assert np.max(np.abs(xyz-d['joints_camera_cv_m']))<1e-6
    front=xyz[:,2]>0; uv=cv2.projectPoints(xyz[front].astype('float64'),np.zeros(3),np.zeros(3),K,dist)[0].reshape(-1,2)
    pixel_error=float(np.max(np.abs(uv-d['joints_2d_px'][front]))); assert pixel_error<1e-3
    assert np.array_equal(ego['joint_names'],names) and np.array_equal(ego['smplx_joint_indices'],idx)
    assert np.allclose(ego['joints_3d_camera_m'],xyz[idx],atol=1e-6)
    assert np.allclose(ego['joints_2d_px'],d['joints_2d_px'][idx],equal_nan=True)
    results[side]={'max_joint_reprojection_error_px':pixel_error,'positive_depth_joints':int(front.sum())}
print(json.dumps({'status':'pass','max_vertex_error_m':vertex_error,'max_joint_error_m':joint_error,'stereo_matrix_max_error':stereo_error,'views':results},indent=2))
