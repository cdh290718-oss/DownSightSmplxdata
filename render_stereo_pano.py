import bpy, json, numpy as np, os, sys, shutil
from mathutils import Matrix
from pathlib import Path
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root))
from rigid_head import cameras_from_head
output_path=os.environ.get('DOWNSIGHT_OUTPUT_DIR');pose_value=os.environ.get('DOWNSIGHT_POSE_PATH');hdr=os.environ.get('DOWNSIGHT_HDR_PATH')
if not output_path or not pose_value or not hdr:raise RuntimeError('Set DOWNSIGHT_OUTPUT_DIR, DOWNSIGHT_POSE_PATH and DOWNSIGHT_HDR_PATH')
out=Path(output_path); out.mkdir(parents=True,exist_ok=True)
s=bpy.context.scene
for im in bpy.data.images:
    if 'smelting-tower_2K_38439fc6' in im.filepath:
        im.filepath=hdr; im.reload()
cam=s.camera
cal=json.loads((root/'calibration'/'stereoParams_20260604_fixed_native.json').read_text(encoding='utf8'))
A=Matrix(cal['poseCamera2']['A']); A.translation=A.translation/1000
mesh=next(o for o in s.objects if o.type=='MESH' and len(o.data.vertices)==10475)
pose_path=Path(pose_value)
rig_path=Path(os.environ.get('DOWNSIGHT_RIG_PATH',str(root/'head_rig_work'/'rig_reference.npz')))
recon=np.load(pose_path);rig=np.load(rig_path)
shutil.copyfile(pose_path,out/'smplx_pose.npz');shutil.copyfile(rig_path,out/'rig_reference.npz')
mw=np.array(mesh.matrix_world,dtype=np.float64)
if not np.allclose(mw,rig['object_matrix_world'],atol=1e-6):raise RuntimeError('Body object transform differs from reference rig')
head_world=mw@recon['head_to_model']
left,right=cameras_from_head(head_world,rig['head_to_left_blender'],rig['head_to_right_blender'])
left_blender,right_blender=Matrix(left.tolist()),Matrix(right.tolist())
mesh.data.vertices.foreach_set('co',recon['vertices_local_m'].reshape(-1)); mesh.data.update()
deps=bpy.context.evaluated_depsgraph_get(); evaluated=mesh.evaluated_get(deps); data=evaluated.to_mesh()
coords=np.empty(len(data.vertices)*3,dtype=np.float32); data.vertices.foreach_get('co',coords); coords=coords.reshape(-1,3)
faces=np.array([list(p.vertices) for p in data.polygons],dtype=np.int32)
mw=np.array(mesh.matrix_world); verts=coords@mw[:3,:3].T+mw[:3,3]
joints=recon['joints_local_m']@mw[:3,:3].T+mw[:3,3]
error=np.linalg.norm(coords-recon['vertices_local_m'],axis=1)
if float(error.max())>1e-6: raise RuntimeError(f'Evaluated mesh differs from SMPL-X: {error.max()}')
np.savez_compressed(out/'geometry.npz',vertices_world_m=verts,joints_world_m=joints,faces=faces,object_matrix_world=mw,head_to_world=head_world,head_to_left_blender=rig['head_to_left_blender'],head_to_right_blender=rig['head_to_right_blender'],left_camera_to_world_blender=np.array(left_blender),right_camera_to_world_blender=np.array(right_blender),left_to_right_cv=np.array(A),max_evaluated_vertex_error_m=error.max())
evaluated.to_mesh_clear()
s.render.engine='CYCLES'; s.cycles.samples=16;s.cycles.use_denoising=True
cam.data.type='PANO';cam.data.panorama_type='EQUIRECTANGULAR'
s.render.resolution_x=4096;s.render.resolution_y=2048;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG'
for name,matrix in [('left',left_blender),('right',right_blender)]:
    cam.matrix_world=matrix;s.render.filepath=str(out/(name+'_pano.png'));bpy.ops.render.render(write_still=True)
