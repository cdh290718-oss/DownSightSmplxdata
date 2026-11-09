"""Extract synchronized body/hand messages from an MCAP without changing it."""
from pathlib import Path
from compression import zstd
import struct,json,os
import numpy as np
from wire import parse

source_path=os.environ.get('DOWNSIGHT_MCAP_PATH')
if not source_path:raise RuntimeError('Set DOWNSIGHT_MCAP_PATH to the MCAP file')
src=Path(source_path);out=Path(os.environ.get('DOWNSIGHT_CONVERSION_DIR',Path(__file__).resolve().parent/'generated_output'/'mcap_conversion'));out.mkdir(parents=True,exist_ok=True)
def st(b,i):n=struct.unpack_from('<I',b,i)[0];return b[i+4:i+4+n].decode(),i+4+n
def fields(b,k):return [v for n,w,v in parse(b) if n==k]
def double(b):return np.frombuffer(b,dtype='<f8').copy()
def vec(b,n):
    v={k:struct.unpack('<d',x)[0] for k,w,x in parse(b) if w==1};return [v.get(i,0.) for i in range(1,n+1)]
def body(data):
    poses=fields(data,3);assert len(poses)==22
    return np.array([vec(fields(p,1)[0],3) for p in poses]),np.array([vec(fields(p,2)[0],4) for p in poses])
def hand(data):
    m=fields(data,7)
    if not m:return None
    v=m[0];representation=fields(v,2)[0].decode(); frame=fields(v,1)[0].decode()
    mats=double(fields(v,5)[0]).reshape(15,3,3)
    bones=fields(data,2);names=[];positions=[]
    for bone in bones:
        name=fields(bone,1);global_pose=fields(bone,3)
        if name and global_pose and fields(global_pose[0],1):
            names.append(name[0].decode());positions.append(vec(fields(global_pose[0],1)[0],3))
    return {'hand_rotmats':mats,'global_rotmat':double(fields(v,4)[0]).reshape(3,3),'transl':double(fields(v,3)[0]),'betas':double(fields(v,6)[0]),'representation':representation,'coordinate_frame':frame,'bone_names':np.array(names),'bone_positions_world':np.array(positions)}
channels={};wanted=[0,15,30,450,900,1350];body_count=0;selected={};hands={'left':{},'right':{}}
with src.open('rb') as f:
    assert f.read(8)==b'\x89MCAP0\r\n'
    while True:
        h=f.read(9)
        if len(h)<9:break
        op,n=struct.unpack('<BQ',h)
        if op!=6:f.seek(n,1);continue
        chunk=f.read(n);compression,i=st(chunk,28);size=struct.unpack_from('<Q',chunk,i)[0];data=chunk[i+8:i+8+size]
        if compression=='zstd':data=zstd.decompress(data)
        elif compression:raise ValueError(compression)
        j=0
        while j+9<=len(data):
            o,length=struct.unpack_from('<BQ',data,j);v=data[j+9:j+9+length];j+=9+length
            if o==4:
                cid,sid=struct.unpack_from('<HH',v);topic,_=st(v,4);channels[cid]=topic
            elif o==5:
                cid,sequence,timestamp,publish=struct.unpack_from('<HIQQ',v);topic=channels.get(cid,'')
                if topic=='robotx/motioncapture/human_pose_fuse_smooth_concat':
                    if body_count in wanted:
                        pos,quat=body(v[22:]);selected[body_count]={'timestamp_ns':timestamp,'body_positions':pos,'body_quaternions_xyzw':quat}
                    body_count+=1
                elif topic in ('/robot0/handtracking/left','/robot0/handtracking/right'):
                    side=topic.rsplit('/',1)[-1]; hnd=hand(v[22:]);
                    if hnd is not None:hands[side][timestamp]=hnd
        if body_count>max(wanted)+3:break
summary=[]
for idx,record in selected.items():
    result={'body_positions_world':record['body_positions'],'body_quaternions_xyzw':record['body_quaternions_xyzw'],'timestamp_ns':np.int64(record['timestamp_ns']),'source_frame_index':np.int32(idx)}; info={'frame':idx,'timestamp_ns':record['timestamp_ns']}
    for side in ('left','right'):
        ts=min(hands[side],key=lambda x:abs(x-record['timestamp_ns']));val=hands[side][ts];info[side+'_time_delta_ms']=(ts-record['timestamp_ns'])/1e6;info[side+'_coordinate_frame']=val['coordinate_frame'];info[side+'_representation']=val['representation']
        if abs(ts-record['timestamp_ns'])>20_000_000:raise ValueError('Body/hand synchronization exceeds 20 ms')
        if val['representation']!='rotmat':raise ValueError('Unexpected hand rotation representation')
        for key in ('hand_rotmats','global_rotmat','transl','betas','bone_names','bone_positions_world'):result[side+'_'+key]=val[key]
    np.savez_compressed(out/f'raw_{idx:04}.npz',**result);summary.append(info)
(out/'extraction.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
