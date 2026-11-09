"""Check saved SMPL-X outputs against the model and fixed head rig."""
from pathlib import Path
import json,os
import numpy as np, torch, smplx
from scipy.spatial.transform import Rotation
from rigid_head import cameras_from_head
root=Path(__file__).resolve().parent; folder=Path(os.environ.get('DOWNSIGHT_CONVERSION_DIR',root/'generated_output'/'mcap_conversion'))
model_dir=os.environ.get('DOWNSIGHT_SMPLX_MODEL_DIR')
if not model_dir:raise RuntimeError('Set DOWNSIGHT_SMPLX_MODEL_DIR to the model directory')
model=smplx.create(model_dir,model_type='smplx',gender='neutral',ext='npz',use_pca=False,flat_hand_mean=True,num_betas=10,num_expression_coeffs=10)
rig=np.load(root/'head_rig_work/rig_reference.npz');reference_pose=np.load(root/'samples/000001/smplx_reconstruction.npz');anchor=reference_pose['joints_local_m'][0];objR=rig['object_matrix_world'][:3,:3];S=np.diag([1,-1,-1,1]); report=[]
fields=['global_orient','transl','body_pose','left_hand_pose','right_hand_pose','betas','expression','jaw_pose','leye_pose','reye_pose']
for path in sorted(folder.glob('smplx_*.npz')):
    p=np.load(path);source=np.load(folder/f"raw_{int(p['source_frame_index']):04}.npz")
    with torch.no_grad():out=model(**{k:torch.from_numpy(p[k].astype('float32')).reshape(1,-1) for k in fields},return_verts=True,return_full_pose=True)
    vert=float(np.max(np.abs(out.vertices[0].numpy()-p['vertices_local_m'])));joint=float(np.max(np.abs(out.joints[0].numpy()-p['joints_local_m'])))
    assert vert<1e-5 and joint<1e-5
    H=rig['object_matrix_world']@p['head_to_model'];L,R=cameras_from_head(H,rig['head_to_left_blender'],rig['head_to_right_blender'])
    stereo=np.linalg.inv(R@S)@(L@S);reference=np.linalg.inv(rig['reference_right_camera_to_world']@S)@(rig['reference_left_camera_to_world']@S)
    stereo_err=float(np.max(np.abs(stereo-reference)));assert stereo_err<1e-6
    rotation_error={};hand_error={};hand_error_scene={}
    hand_ids={'left':[20,37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70], 'right':[21,52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75]}
    for side in ('left','right'):
        raw=source[side+'_hand_rotmats'];F=np.diag([-1.,1.,1.]) if side=='left' else np.eye(3);expected=F@raw@F
        converted=Rotation.from_rotvec(p[side+'_hand_pose'].reshape(-1,3)).as_matrix();rotation_error[side]=float(np.max(np.abs(expected-converted)));assert rotation_error[side]<1e-5
        x=p['joints_local_m'][hand_ids[side]].astype(float);y=source[side+'_bone_positions_world'].copy();x-=x.mean(0);y-=y.mean(0)
        u,_,v=np.linalg.svd(x.T@y);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@v);hand_error[side]=float(np.linalg.norm(x@u@fix@v-y,axis=1).mean()*1000)
        target=(source[side+'_bone_positions_world']-source['body_positions_world'][0])@objR+anchor
        hand_error_scene[side]=float(np.linalg.norm(p['joints_local_m'][hand_ids[side]]-target,axis=1).mean()*1000)
    report.append({'frame':int(p['source_frame_index']),'max_vertex_difference_m':vert,'max_joint_difference_m':joint,'max_stereo_difference':stereo_err,'hand_matrix_roundtrip_max':rotation_error,'hand_joint_mean_mm_after_separate_rigid_alignment':hand_error,'hand_joint_mean_mm_body_aligned':hand_error_scene})
(folder/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
