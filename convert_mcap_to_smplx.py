"""Experimental 22-joint IK + MANO rotations; produces renderable SMPL-X poses."""
from pathlib import Path
import json,time,os
import numpy as np, torch, smplx
from scipy.spatial.transform import Rotation

root=Path(__file__).resolve().parent; out=Path(os.environ.get('DOWNSIGHT_CONVERSION_DIR',root/'generated_output'/'mcap_conversion'));files=sorted(out.glob('raw_*.npz'))
if not files: raise RuntimeError('Run mcap_extract_motion.py first')
data=[np.load(p,allow_pickle=False) for p in files]; n=len(data)
ref=np.load(root/'samples/000001/smplx_reconstruction.npz');rig=np.load(root/'head_rig_work/rig_reference.npz')
device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
model_dir=os.environ.get('DOWNSIGHT_SMPLX_MODEL_DIR')
if not model_dir:raise RuntimeError('Set DOWNSIGHT_SMPLX_MODEL_DIR to the directory containing smplx/SMPLX_NEUTRAL.npz')
model=smplx.create(model_dir,model_type='smplx',gender='neutral',ext='npz',use_pca=False,flat_hand_mean=True,num_betas=10,num_expression_coeffs=10).to(device)
Robj=rig['object_matrix_world'][:3,:3]; anchor=ref['joints_local_m'][0]
targets=np.stack([(d['body_positions_world']-d['body_positions_world'][0])@Robj+anchor for d in data]).astype('float32')
to=lambda a:torch.tensor(a,device=device,dtype=torch.float32)
target=to(targets);betas=to(np.tile(ref['betas'],(n,1))); hand_values={};hand_metrics={}
face={'jaw_pose':torch.zeros(n,3,device=device),'leye_pose':torch.zeros(n,3,device=device),'reye_pose':torch.zeros(n,3,device=device),'expression':torch.zeros(n,10,device=device)}
for side in ('left','right'):
    matrices=np.stack([d[side+'_hand_rotmats'] for d in data]);rt=matrices.transpose(0,1,3,2)@matrices
    orth=float(np.abs(rt-np.eye(3)).max());det=float(np.abs(np.linalg.det(matrices)-1).max())
    if orth>1e-3 or det>1e-3:raise ValueError('Invalid '+side+' MANO rotation matrices')
    # The source left MANO rotations use the opposite handed local basis.
    # For an improper basis reflection F, a rotation changes as F @ R @ F.
    F=np.diag([-1.,1.,1.]) if side=='left' else np.eye(3)
    converted=F@matrices@F
    rv=Rotation.from_matrix(converted.reshape(-1,3,3)).as_rotvec().reshape(n,45).astype('float32')
    restored=Rotation.from_rotvec(rv.reshape(-1,3)).as_matrix().reshape(n,15,3,3)
    hand_metrics[side]={'orthogonality_max':orth,'det_error_max':det,'rotvec_roundtrip_max':float(np.max(np.abs(restored-converted))),'basis_conversion':'F R F with F=diag(-1,1,1)' if side=='left' else 'identity','finger_order':'assumed MANO native index-middle-pinky-ring-thumb; independently compare hand keypoints'}
    hand_values[side]=to(rv)
pose=torch.nn.Parameter(to(np.tile(ref['body_pose'],(n,1)))); glob=torch.nn.Parameter(to(np.tile(ref['global_orient'],(n,1)))); transl=torch.nn.Parameter(torch.zeros(n,3,device=device))
optimizer=torch.optim.Adam([pose,glob,transl],lr=.035)
weights=to([1.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,3]).reshape(1,22,1)
start=time.time()
for step in range(400):
    optimizer.zero_grad(); p=pose.reshape(n,21,3)
    # Joint 15 (head) orientation is not identifiable from joint positions.
    body=torch.cat((p[:,:14,:],torch.zeros_like(p[:,14:15,:]),p[:,15:,:]),dim=1).reshape(n,63)
    result=model(global_orient=glob,body_pose=body,transl=transl,betas=betas,left_hand_pose=hand_values['left'],right_hand_pose=hand_values['right'],**face)
    residual=result.joints[:,:22]-target
    loss=((residual*weights)**2).mean()+0.00006*(body**2).mean()
    loss.backward();optimizer.step()
    if step%100==0:print('step',step,'loss',float(loss.detach()),flush=True)
pose_np=pose.detach().cpu().numpy().reshape(n,21,3);pose_np[:,14]=0
params={'global_orient':glob.detach().cpu().numpy(),'transl':transl.detach().cpu().numpy(),'body_pose':pose_np.reshape(n,63),'betas':betas.cpu().numpy(),'left_hand_pose':hand_values['left'].cpu().numpy(),'right_hand_pose':hand_values['right'].cpu().numpy(),**{k:v.cpu().numpy() for k,v in face.items()}}
parents=model.parents.cpu().numpy();chain=[];j=15
while j>=0:chain.append(int(j));j=int(parents[j])
with torch.no_grad(): preliminary=model(**{k:to(v) for k,v in params.items()},return_verts=True,return_full_pose=True)
hand_ids={'left':[20,37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70], 'right':[21,52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75]}
for i,d in enumerate(data):
    full=preliminary.full_pose[i].cpu().numpy().reshape(-1,3)
    for side,wrist in [('left',20),('right',21)]:
        # Align the predicted hand to tracked landmarks in model coordinates;
        # fit orientation after centering so wrist-position error is not hidden.
        x=preliminary.joints[i,hand_ids[side]].cpu().numpy().astype(float)
        y=(d[side+'_bone_positions_world']-d['body_positions_world'][0])@Robj+anchor
        x-=x.mean(0);y-=y.mean(0);u,_,v=np.linalg.svd(x.T@y);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@v)
        delta=(u@fix@v).T
        parent_chain=[];k=int(parents[wrist])
        while k>=0:parent_chain.append(k);k=int(parents[k])
        parent_rot=np.eye(3)
        for k in reversed(parent_chain):parent_rot=parent_rot@Rotation.from_rotvec(full[k]).as_matrix()
        old=Rotation.from_rotvec(full[wrist]).as_matrix()
        params['body_pose'][i].reshape(21,3)[wrist-1]=Rotation.from_matrix(parent_rot.T@delta@parent_rot@old).as_rotvec()
with torch.no_grad():final=model(**{k:to(v) for k,v in params.items()},return_verts=True,return_full_pose=True)
reports=[]
for i,d in enumerate(data):
    joints=final.joints[i].cpu().numpy();verts=final.vertices[i].cpu().numpy();full=final.full_pose[i].cpu().numpy().reshape(-1,3)
    rot=np.eye(3)
    for j in reversed(chain):rot=rot@Rotation.from_rotvec(full[j]).as_matrix()
    H=np.eye(4);H[:3,:3]=rot;H[:3,3]=joints[15]
    err=np.linalg.norm(joints[:22]-targets[i],axis=1);idx=int(d['source_frame_index']); posefile=out/f'smplx_{idx:04}.npz'
    np.savez_compressed(posefile,vertices_local_m=verts,joints_local_m=joints,head_to_model=H,head_chain=np.array(chain),source_frame_index=idx,source_timestamp_ns=d['timestamp_ns'],target_joints_model_m=targets[i],flat_hand_mean=np.array(True),**{k:v[i] for k,v in params.items()})
    reports.append({'frame':idx,'fit_mean_error_m':float(err.mean()),'fit_max_error_m':float(err.max()),'per_joint_error_m':err.tolist(),'pose_file':posefile.name})
(out/'fit_report.json').write_text(json.dumps({'device':device,'seconds':time.time()-start,'joint_order':'assumed SMPL-X 22-joint order, inferred from spatial structure; not documented','head_local_rotation':'zero: not observable from joint positions','wrist_orientation':'estimated by rigid alignment of each hand to its tracked 21 landmarks; translation not optimized','hands':hand_metrics,'frames':reports},indent=2),encoding='utf8')
print(json.dumps(reports,indent=2))
