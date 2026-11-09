"""Remap Cycles equirectangular renders into the source Brown-Conrady cameras."""
from pathlib import Path
import json,os
import cv2
import numpy as np

root=Path(__file__).resolve().parent; output_path=os.environ.get('DOWNSIGHT_OUTPUT_DIR')
if not output_path:raise RuntimeError('Set DOWNSIGHT_OUTPUT_DIR to the rendered panorama directory')
out=Path(output_path)
cal=json.loads((root/'calibration'/'stereoParams_20260604_fixed_native.json').read_text(encoding='utf8'))
geo=np.load(out/'geometry.npz'); vertices=geo['vertices_world_m']; S=np.diag([1.,-1.,-1.,1.])
results={}
for name,key in [('left','camera1'),('right','camera2')]:
    c=cal[key]; K=np.array(c['K'],dtype=np.float64); K[:2,2]-=1.0 # MATLAB 1-based to OpenCV 0-based
    dist=np.array([*c['radialDistortion'],*c['tangentialDistortion'],0.],dtype=np.float64)
    h,w=c['imageSize']; yy,xx=np.indices((h,w),dtype=np.float64); pix=np.stack((xx,yy),axis=-1).reshape(-1,1,2)
    assert np.allclose(dist[2:],0), 'Only radial k1,k2 are supported'
    xd=(xx-K[0,2])/K[0,0]; yd=(yy-K[1,2])/K[1,1]; rd=np.hypot(xd,yd); r=rd.copy()
    for _ in range(20):
        r2=r*r; r-=(r*(1+dist[0]*r2+dist[1]*r2*r2)-rd)/(1+3*dist[0]*r2+5*dist[1]*r2*r2)
    scale=np.divide(r,rd,out=np.ones_like(r),where=rd>0); rays=np.stack((xd*scale,yd*scale),axis=-1)
    xyz=np.concatenate([rays.reshape(-1,2),np.ones((w*h,1))],axis=1)
    recon,_=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,dist); err=np.linalg.norm(recon.reshape(-1,2)-pix.reshape(-1,2),axis=1)
    if err.max()>0.01: raise RuntimeError(f'Undistortion did not converge: {name} {err.max()}')
    pano=cv2.imdecode(np.fromfile(out/(name+'_pano.png'),dtype=np.uint8),cv2.IMREAD_COLOR)
    ph,pw=pano.shape[:2]; x=rays[:,:,0]; y=rays[:,:,1]
    mx=((.5+np.arctan2(x,1)/(2*np.pi))*pw-.5).astype(np.float32)
    my=((.5+np.arctan2(y,np.sqrt(1+x*x))/np.pi)*ph-.5).astype(np.float32)
    image=cv2.remap(pano,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP)
    cv2.imencode('.png',image)[1].tofile(out/(name+'.png'))
    wc=np.linalg.inv(geo[name+'_camera_to_world_blender']@S)
    p=vertices@wc[:3,:3].T+wc[:3,3]; uv=np.full((len(p),2),np.nan); positive=p[:,2]>0
    uv[positive]=cv2.projectPoints(p[positive],np.zeros(3),np.zeros(3),K,dist)[0].reshape(-1,2)
    in_frame=positive & (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    j=geo['joints_world_m']@wc[:3,:3].T+wc[:3,3]; juv=np.full((len(j),2),np.nan); jp=j[:,2]>0
    juv[jp]=cv2.projectPoints(j[jp],np.zeros(3),np.zeros(3),K,dist)[0].reshape(-1,2)
    jin=jp & (juv[:,0]>=0)&(juv[:,0]<w)&(juv[:,1]>=0)&(juv[:,1]<h)
    np.savez_compressed(out/(name+'_vertices.npz'),vertices_camera_cv_m=p,vertices_2d_px=uv,in_frame_no_occlusion_test=in_frame,joints_camera_cv_m=j,joints_2d_px=juv,joints_in_frame_no_occlusion_test=jin,K_zero_based=K,distortion=dist,world_to_camera_cv=wc)
    results[name]={'inverse_projection_mean_error_px':float(err.mean()),'inverse_projection_max_error_px':float(err.max()),'vertex_in_frame_count':int(in_frame.sum()),'render_size':[w,h]}
(out/'validation.json').write_text(json.dumps(results,indent=2),encoding='utf8')
print(json.dumps(results,indent=2))
