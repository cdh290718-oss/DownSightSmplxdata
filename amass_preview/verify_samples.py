"""Validate saved AMASS preview geometry and camera projections."""
from pathlib import Path
import json
import cv2
import numpy as np

root=Path(__file__).resolve().parent
results={}
for folder in sorted((root/'samples').iterdir()):
    if not folder.is_dir(): continue
    pose=np.load(folder/'smplx_pose.npz',allow_pickle=False)
    geo=np.load(folder/'geometry.npz',allow_pickle=False)
    swap=np.diag([1.,-1.,-1.,1.])
    left=geo['left_camera_to_world_blender']@swap
    right=geo['right_camera_to_world_blender']@swap
    error=float(np.max(np.abs(np.linalg.inv(right)@left-geo['left_to_right_cv'])))
    assert error<1e-5,(folder.name,error)
    assert np.allclose(pose['vertices_local_m']@geo['object_matrix_world'][:3,:3].T+geo['object_matrix_world'][:3,3],geo['vertices_world_m'],atol=1e-6)
    for side in ('left','right'):
        im=cv2.imdecode(np.fromfile(folder/(side+'.png'),np.uint8),cv2.IMREAD_COLOR)
        assert im is not None and im.shape==(720,1280,3),folder/side
        v=np.load(folder/(side+'_vertices.npz'),allow_pickle=False)
        points=pose['joints_local_m']@geo['object_matrix_world'][:3,:3].T+geo['object_matrix_world'][:3,3]
        pts=points@v['world_to_camera_cv'][:3,:3].T+v['world_to_camera_cv'][:3,3]
        assert np.allclose(pts,v['joints_camera_cv_m'],atol=1e-6)
        front=pts[:,2]>0
        proj=cv2.projectPoints(pts[front],np.zeros(3),np.zeros(3),v['K_zero_based'],v['distortion'])[0].reshape(-1,2)
        assert np.allclose(proj,v['joints_2d_px'][front],atol=1e-3)
    results[folder.name]={'stereo_transform_max_error':error,'baseline_m':float(np.linalg.norm((np.linalg.inv(right)@left)[:3,3])),'source_frame_index':int(pose['source_frame_index'])}
assert len(results)==6
print(json.dumps({'status':'pass','samples':results},indent=2))
