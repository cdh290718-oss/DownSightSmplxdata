from pathlib import Path
import json,unittest
import numpy as np
from scipy.spatial.transform import Rotation
from rigid_head import cameras_from_head

root=Path(__file__).resolve().parent/'head_rig_work'
class HeadRigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rig=np.load(root/'rig_reference.npz');cls.ref=np.load(root/'reference_pose.npz');cls.yaw=np.load(root/'head_yaw25.npz')
    def cameras(self,H):return cameras_from_head(H,self.rig['head_to_left_blender'],self.rig['head_to_right_blender'])
    def test_reference_is_unchanged(self):
        H=self.rig['object_matrix_world']@self.ref['head_to_model'];L,R=self.cameras(H)
        np.testing.assert_allclose(L,self.rig['reference_left_camera_to_world'],atol=1e-7)
        np.testing.assert_allclose(R,self.rig['reference_right_camera_to_world'],atol=1e-7)
    def test_actual_smplx_head_turn_moves_both_cameras(self):
        H0=self.rig['object_matrix_world']@self.ref['head_to_model'];H1=self.rig['object_matrix_world']@self.yaw['head_to_model']
        L0,R0=self.cameras(H0);L1,R1=self.cameras(H1)
        angle=Rotation.from_matrix(H0[:3,:3].T@H1[:3,:3]).magnitude()
        self.assertAlmostEqual(np.degrees(angle),25,places=4)
        np.testing.assert_allclose(np.linalg.inv(H0)@L0,np.linalg.inv(H1)@L1,atol=1e-9)
        np.testing.assert_allclose(np.linalg.inv(H0)@R0,np.linalg.inv(H1)@R1,atol=1e-9)
        np.testing.assert_allclose(np.linalg.inv(R0)@L0,np.linalg.inv(R1)@L1,atol=1e-9)
        self.assertGreater(np.linalg.norm(L1[:3,3]-L0[:3,3]),0.01)
    def test_rigid_world_motion_is_equivariant(self):
        H=self.rig['reference_head_to_world'];L,R=self.cameras(H); G=np.eye(4);G[:3,:3]=Rotation.from_euler('xyz',[20,15,30],degrees=True).as_matrix();G[:3,3]=[.4,-.2,1.3]
        newL,newR=self.cameras(G@H);np.testing.assert_allclose(newL,G@L,atol=1e-9);np.testing.assert_allclose(newR,G@R,atol=1e-9)

if __name__=='__main__':unittest.main(verbosity=2)
