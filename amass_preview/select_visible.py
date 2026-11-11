from pathlib import Path
import tarfile,io,os,numpy as np,torch,smplx,json,cv2
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parent.parent; dataset=os.environ.get('DOWNSIGHT_AMASS_DATASET','ACCAD').upper(); assert dataset in ('ACCAD','CNRS'); archive=Path(os.environ['DOWNSIGHT_AMASS_ARCHIVE']); model_dir=os.environ['DOWNSIGHT_SMPLX_MODEL_DIR']; out=Path(os.environ.get('DOWNSIGHT_OUTPUT_DIR',str(root/'generated_output'/'amass_preview'/dataset))); out.mkdir(parents=True,exist_ok=True)
folders=['Female1General_c3d','Male1General_c3d','Female1Gestures_c3d','Male2General_c3d','Female1Walking_c3d','Male2Walking_c3d'] if dataset=='ACCAD' else ['283','288']; limit=3 if dataset=='ACCAD' else 8
selected={}; files={}; per={k:0 for k in folders}
with tarfile.open(archive,'r|bz2') as t:
 for m in t:
  x=m.name.split('/')
  if len(x)!=3 or x[1] not in per or not m.name.endswith('_stageii.npz') or per[x[1]]>=limit:continue
  if any(x in m.name.lower() for x in ['subject_cal','stagei.npz','_cal_']):continue
  z=np.load(io.BytesIO(t.extractfile(m).read()),allow_pickle=False)
  d={k:z[k] for k in ['gender','trans','betas','root_orient','pose_body','pose_hand','pose_jaw','pose_eye','mocap_frame_rate']};files[m.name]=d;per[x[1]]+=1
print('Selected sequences',len(files),per,flush=True)
cal=json.loads((root/'calibration/stereoParams_20260604_fixed_native.json').read_text(encoding='utf8'));rig=np.load(root/'head_rig_work/rig_reference.npz');ref=np.load(root/'head_rig_work/reference_pose.npz');obj=rig['object_matrix_world'];S=np.diag([1.,-1.,-1.,1.]);target=ref['joints_local_m'][0];to=lambda a:torch.as_tensor(np.asarray(a,dtype=np.float32));torch.set_num_threads(4)
models={}; candidates=[]; relevant=np.array([16,17,18,19,20,21,4,5,7,8]); headid=15
for path,d in files.items():
 n=len(d['trans']); gender=str(d['gender']); ids=np.unique(np.linspace(1,max(1,n-2),min(35,max(1,n-2))).astype(int));B=len(ids)
 if gender not in models:models[gender]=smplx.create(model_dir,model_type='smplx',gender=gender,ext='npz',use_pca=False,flat_hand_mean=True,num_betas=16,num_expression_coeffs=10)
 model=models[gender]; glob=(Rotation.from_euler('x',-90,degrees=True)*Rotation.from_rotvec(d['root_orient'][ids])).as_rotvec()
 params={'betas':np.tile(d['betas'],(B,1)),'transl':np.zeros((B,3)),'global_orient':glob,'body_pose':d['pose_body'][ids],'left_hand_pose':d['pose_hand'][ids,:45],'right_hand_pose':d['pose_hand'][ids,45:],'jaw_pose':d['pose_jaw'][ids],'leye_pose':d['pose_eye'][ids,:3],'reye_pose':d['pose_eye'][ids,3:],'expression':np.zeros((B,10))}
 with torch.no_grad():result=model(**{k:to(v) for k,v in params.items()},return_verts=True,return_full_pose=True)
 j=result.joints.numpy();verts=result.vertices.numpy(); full=result.full_pose.numpy().reshape(B,-1,3);chain=[];p=headid
 while p>=0:chain.append(int(p));p=int(model.parents[p])
 for a,idx in enumerate(ids):
  H=np.eye(4)
  for k in reversed(chain):H[:3,:3]=H[:3,:3]@Rotation.from_rotvec(full[a,k]).as_matrix()
  shift=target-j[a,0];H[:3,3]=j[a,headid]+shift
  score=0; metrics={}; ok=True
  for side,key in [('left','camera1'),('right','camera2')]:
   cam=obj@H@rig['head_to_'+side+'_blender']; mat=np.linalg.inv(cam@S)@obj; xyz=(j[a,relevant]+shift)@mat[:3,:3].T+mat[:3,3];K=np.array(cal[key]['K']);K[:2,2]-=1;dist=np.array([*cal[key]['radialDistortion'],*cal[key]['tangentialDistortion'],0.]);uv=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,dist)[0].reshape(-1,2)
   valid=(xyz[:,2]>0)&(uv[:,0]>=30)&(uv[:,0]<1250)&(uv[:,1]>=30)&(uv[:,1]<690); wrists=valid[4:6].sum();upper=valid[:6].sum();legs=valid[6:].sum(); metrics[side]={'upper':int(upper),'wrists':int(wrists),'legs':int(legs),'uv':uv.tolist(),'valid':valid.tolist()};score+=int(wrists)*10+int(upper)*2+int(legs)*3;ok &= wrists==2 and legs>=3
  speed=float(np.linalg.norm(d['pose_body'][idx+1]-d['pose_body'][idx-1]));score-=min(speed,5)*2
  if ok:candidates.append({'score':float(score),'speed_proxy':speed,'source':path,'frame':int(idx),'gender':gender,'metrics':metrics,'data':d,'params':{k:v[a] for k,v in params.items()},'vertices':verts[a]+shift,'joints':j[a]+shift,'head':H,'chain':chain})
 print('evaluated',path,flush=True)
exec((root/'amass_preview'/'visibility_rank.py').read_text(encoding='utf8'))
candidates.sort(key=lambda x:x['visibility_score'],reverse=True)
print('matching',len(candidates),flush=True)
used=set(); picked=[]
for c in candidates:
 if c['source'] in used:continue
 used.add(c['source']);picked.append(c)
 if len(picked)==3:break
report=[]
for i,c in enumerate(picked,1):
 folder=out/f'{i:02d}';folder.mkdir(exist_ok=True);par=c['params'];par['transl']=target-c['joints'][0]+par['transl']
 # joints and vertices are shifted; set translation to match the saved geometry
 orig=files[c['source']];z=np.load(root/'head_rig_work/reference_pose.npz');
 # shift was applied to the zero-translation model; recover it by comparing raw model pelvis
 model=models[c['gender']]
 with torch.no_grad():r=model(**{k:to(np.asarray(v).reshape(1,-1)) for k,v in c['params'].items()},return_verts=True)
 par['transl']=target-r.joints[0,0].numpy()
 np.savez_compressed(folder/'smplx_pose.npz',vertices_local_m=c['vertices'],joints_local_m=c['joints'],head_to_model=c['head'],head_chain=np.array(c['chain']),source_frame_index=c['frame'],gender=np.array(c['gender']),flat_hand_mean=np.array(True),**par)
 report.append({k:v for k,v in c.items() if k not in ['data','params','vertices','joints','head','chain']})
(out/'selection.json').write_text(json.dumps({'sampled_sequences':len(files),'candidates':len(candidates),'selection':report},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'sampled_sequences':len(files),'candidates':len(candidates),'selection':report},ensure_ascii=False,indent=2),flush=True)
