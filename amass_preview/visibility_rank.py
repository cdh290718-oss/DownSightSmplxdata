candidates.sort(key=lambda x:x['score'],reverse=True)
subset=[]
for src in dict.fromkeys(c['source'] for c in candidates):subset.extend([c for c in candidates if c['source']==src][:15])
print('Checking surface visibility on',len(subset),'frames',flush=True)
seg={}
for gender,model in models.items():
 w=model.lbs_weights.detach().cpu().numpy();groups=[[0,3,6,9,12,15,22,23,24],[13,16,18,20]+list(range(25,40)),[14,17,19,21]+list(range(40,55)),[1,4],[2,5],[7,10],[8,11]]
 labels=np.argmax(np.stack([w[:,x].sum(1) for x in groups],axis=1),axis=1).astype(np.uint8);faces=model.faces.astype(np.int32)
 tri=np.array([np.bincount(v,minlength=7).argmax() for v in labels[faces]],dtype=np.uint8);seg[gender]=(faces,tri)
for ix,c in enumerate(subset):
 faces,labels=seg[c['gender']];allcounts=[]
 for side,key in [('left','camera1'),('right','camera2')]:
  cam=obj@c['head']@rig['head_to_'+side+'_blender'];T=np.linalg.inv(cam@S)@obj;xyz=c['vertices']@T[:3,:3].T+T[:3,3];K=np.array(cal[key]['K']);K[:2,2]-=1;dist=np.array([*cal[key]['radialDistortion'],*cal[key]['tangentialDistortion'],0.]);uv=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,dist)[0].reshape(-1,2)*0.25
  mask=(xyz[faces,2]>0.08).all(axis=1);vv=faces[mask];lab=labels[mask];coords=uv[vv];good=(coords.max(axis=(1,2))<5000)&(coords.min(axis=(1,2))>-5000)&(coords[:,:,0].max(1)>=0)&(coords[:,:,0].min(1)<320)&(coords[:,:,1].max(1)>=0)&(coords[:,:,1].min(1)<180);vv=vv[good];lab=lab[good];coords=coords[good];order=np.argsort(xyz[vv,2].mean(axis=1))[::-1];canvas=np.zeros((180,320),np.uint8)
  for i in order:cv2.fillConvexPoly(canvas,np.rint(coords[i]).astype(np.int32),int(lab[i]+1))
  count=np.bincount(canvas.reshape(-1),minlength=8);allcounts.append([int(count[k+1]) for k in [1,2,3,4]])
 c['surface_pixels']=allcounts;weakest=min(min(a) for a in allcounts);c['visibility_score']=float(weakest+0.03*sum(sum(a) for a in allcounts)-c['speed_proxy']*5)
 if ix%20==0:print('visibility',ix,'/',len(subset),flush=True)
candidates=subset
