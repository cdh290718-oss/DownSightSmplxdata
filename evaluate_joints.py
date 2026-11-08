"""Compare camera-frame EgoSMPLX predictions against the synthetic sample."""
import argparse, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--prediction',type=Path,required=True)
p.add_argument('--side',choices=['left','right'],required=True)
p.add_argument('--sample',type=Path,default=Path(__file__).parent/'samples/000001')
a=p.parse_args()
gt=np.load(a.sample/(a.side+'_egosmplx65.npz'),allow_pickle=False)
pred=np.load(a.prediction,allow_pickle=False)
if 'smplx_body_joints_3d_render_cam' in pred:
    key='smplx_body_joints_3d_render_cam'; count=25
elif 'joints_3d_camera_m' in pred:
    key='joints_3d_camera_m'; count=65
else:
    raise ValueError('Prediction must contain smplx_body_joints_3d_render_cam (25) or joints_3d_camera_m (65)')
q=np.asarray(pred[key],dtype=float).reshape(-1,3)
if len(q)!=count: raise ValueError(f'{key} must have {count} points, got {len(q)}')
if not np.isfinite(q).all(): raise ValueError('Prediction contains non-finite values')
truth=gt['joints_3d_camera_m'][:count]; names=gt['joint_names'][:count]
absolute=np.linalg.norm(q-truth,axis=1)*1000
root_relative=np.linalg.norm((q-q[0])-(truth-truth[0]),axis=1)*1000
print(json.dumps({'side':a.side,'prediction_field':key,'joint_count':count,'absolute_mean_mm':float(absolute.mean()),'pelvis_aligned_mean_mm':float(root_relative.mean()),'per_joint':[{'name':str(n),'absolute_mm':float(x),'pelvis_aligned_mm':float(y)} for n,x,y in zip(names,absolute,root_relative)]},indent=2))
