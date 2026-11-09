"""Rigid camera mounting; transforms map column-vector coordinates to parent."""
import numpy as np

def validate_rigid(T, name):
    T=np.asarray(T,dtype=np.float64)
    if T.shape!=(4,4) or not np.isfinite(T).all(): raise ValueError(name+' must be a finite 4x4 matrix')
    R=T[:3,:3]
    if not np.allclose(T[3], [0,0,0,1],atol=1e-6) or not np.allclose(R.T@R,np.eye(3),atol=1e-5) or abs(np.linalg.det(R)-1)>1e-5:
        raise ValueError(name+' is not a rigid transform')
    return T

def camera_mount(head_world,left_world,right_world):
    h=validate_rigid(head_world,'head_world'); l=validate_rigid(left_world,'left_world'); r=validate_rigid(right_world,'right_world')
    return np.linalg.inv(h)@l,np.linalg.inv(h)@r

def cameras_from_head(head_world,head_to_left,head_to_right):
    h=validate_rigid(head_world,'head_world')
    return h@validate_rigid(head_to_left,'head_to_left'),h@validate_rigid(head_to_right,'head_to_right')
