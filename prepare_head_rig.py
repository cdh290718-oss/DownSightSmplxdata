"""Derive posed head transforms from actual SMPL-X kinematic chain."""
from pathlib import Path
import argparse, json, os
import numpy as np
import torch,smplx
from scipy.spatial.transform import Rotation
from rigid_head import camera_mount,cameras_from_head

p=argparse.ArgumentParser();p.add_argument('--head-yaw-deg',type=float,default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--init-rig',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parent; sample=root/'samples/000001'; source=np.load(sample/'smplx_reconstruction.npz',allow_pickle=False)
model_dir=os.environ.get('DOWNSIGHT_SMPLX_MODEL_DIR')
if not model_dir:raise RuntimeError('Set DOWNSIGHT_SMPLX_MODEL_DIR to the model directory')
model=smplx.create(model_dir,model_type='smplx',gender='neutral',ext='npz',use_pca=False,num_betas=10,num_expression_coeffs=10)
fields=['global_orient','transl','body_pose','left_hand_pose','right_hand_pose','betas','expression','jaw_pose','leye_pose','reye_pose']
parameters={k:source[k].copy() for k in fields}
if args.head_yaw_deg:
    # Head joint 15 corresponds to body_pose element 14 after root joint 0.
    pose=parameters['body_pose'].reshape(21,3).copy(); pose[14]= (Rotation.from_rotvec(pose[14])*Rotation.from_euler('y',args.head_yaw_deg,degrees=True)).as_rotvec();parameters['body_pose']=pose.reshape(-1)
with torch.no_grad(): result=model(**{k:torch.from_numpy(v.astype('float32')).reshape(1,-1) for k,v in parameters.items()},return_verts=True,return_full_pose=True)
joints=result.joints[0].numpy(); vertices=result.vertices[0].numpy(); full=result.full_pose[0].numpy().reshape(-1,3); parents=model.parents.cpu().numpy()
head=15; chain=[]; k=head
while k>=0: chain.append(int(k));k=int(parents[k])
rotation=np.eye(3)
for k in reversed(chain): rotation=rotation@Rotation.from_rotvec(full[k]).as_matrix()
H=np.eye(4);H[:3,:3]=rotation;H[:3,3]=joints[head]
args.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(args.output,vertices_local_m=vertices,joints_local_m=joints,head_to_model=H,head_chain=np.array(chain),**parameters)
if args.init_rig:
    if args.head_yaw_deg: raise ValueError('Rig must be initialized from unchanged reference pose')
    g=np.load(sample/'geometry.npz'); obj=g['object_matrix_world']; head_world=obj@H
    left,right=camera_mount(head_world,g['left_camera_to_world_blender'],g['right_camera_to_world_blender'])
    rig=args.output.parent/'rig_reference.npz'
    if rig.exists(): raise FileExistsError(rig)
    np.savez_compressed(rig,head_to_left_blender=left,head_to_right_blender=right,reference_head_to_world=head_world,object_matrix_world=obj,reference_left_camera_to_world=g['left_camera_to_world_blender'],reference_right_camera_to_world=g['right_camera_to_world_blender'])
    l,r=cameras_from_head(head_world,left,right)
    print(json.dumps({'head_chain_head_to_root':chain,'max_reference_camera_error':max(float(np.abs(l-g['left_camera_to_world_blender']).max()),float(np.abs(r-g['right_camera_to_world_blender']).max())),'rig':str(rig)},indent=2))
else:print('saved',args.output)
