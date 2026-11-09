"""Compare candidate left MANO conventions using 3D tracked hand landmarks."""
from pathlib import Path
import numpy as np,torch,smplx,json,os
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parent;folder=Path(os.environ.get('DOWNSIGHT_CONVERSION_DIR',root/'generated_output'/'mcap_conversion'));d=np.load(folder/'smplx_0450.npz');raw=np.load(folder/'raw_0450.npz')
model_dir=os.environ.get('DOWNSIGHT_SMPLX_MODEL_DIR')
if not model_dir:raise RuntimeError('Set DOWNSIGHT_SMPLX_MODEL_DIR to the model directory')
model=smplx.create(model_dir,model_type='smplx',gender='neutral',ext='npz',use_pca=False,flat_hand_mean=True,num_betas=10,num_expression_coeffs=10)
base=raw['left_hand_rotmats'];variants={}
for transpose in [False,True]:
    mats=base.transpose(0,2,1) if transpose else base
    for axis in ['none','x','y','z']:
        flip=np.eye(3)
        if axis!='none':flip['xyz'.index(axis),'xyz'.index(axis)]=-1
        variants[f'transpose={transpose},reflection={axis}']=flip@mats@flip
fields=['global_orient','transl','body_pose','right_hand_pose','betas','expression','jaw_pose','leye_pose','reye_pose'];N=len(variants)
kw={k:torch.tensor(np.tile(d[k],(N,1)),dtype=torch.float32) for k in fields}
kw['left_hand_pose']=torch.tensor(np.stack([Rotation.from_matrix(v).as_rotvec().reshape(45) for v in variants.values()]),dtype=torch.float32)
with torch.no_grad():o=model(**kw)
ids=[20,37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70];target=raw['left_bone_positions_world'];target=target-target.mean(0);results={}
for i,name in enumerate(variants):
    x=o.joints[i,ids].numpy().astype(float);x-=x.mean(0);u,_,v=np.linalg.svd(x.T@target);fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@v);q=u@fix@v
    results[name]=float(np.linalg.norm(x@q-target,axis=1).mean()*1000)
(folder/'left_hand_conventions.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
