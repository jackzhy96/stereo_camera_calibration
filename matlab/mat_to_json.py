import os
import sys
import scipy.io
import json
dynamic_path = os.path.abspath(__file__+"/../")
# print(dynamic_path)
sys.path.append(dynamic_path)

def mat_parser(mat_path, data_path):
    mat = scipy.io.loadmat(mat_path)
    cam1_intrinsics = mat['intrinsicMatrix1'].tolist()
    cam2_intrinsics = mat['intrinsicMatrix2'].tolist()
    cam1_distortion = mat['distortionCoefficients1'].tolist()
    cam2_distortion = mat['distortionCoefficients2'].tolist()
    rot_cam2 = mat['rotationOfCamera2'].tolist()
    pos_cam2 = mat['translationOfCamera2'].tolist()
    calib_result = {
        'cam1_intrinsics': cam1_intrinsics,
        'cam2_intrinsics': cam2_intrinsics,
        'cam1_distortion': cam1_distortion,
        'cam2_distortion': cam2_distortion,
        'rot_cam2': rot_cam2,
        'pos_cam2': pos_cam2
    }
    json_w = json.dumps(calib_result)
    f = open(data_path, 'w')
    f.write(json_w)
    f.close()

if __name__ == "__main__":
    mat_path = os.path.join(dynamic_path, "calib_result.mat")
    data_path = os.path.join(dynamic_path, "result_calibration.json")
    mat_parser(mat_path, data_path)
    print('Done!')




    # traj = mat['q']  # (3, point_num)
    # jaw_angle = mat['angle']
    # last_js = mat['last_joint']
    # wrist_js = mat['wrist_joint']
    # shaft_js = mat['shaft_joint']
    # print(f"trajectory shape: {traj.shape}")
    # print(f"jaw opening angle shape: {jaw_angle.shape}")
    # print(f"last joint state shape: {last_js.shape}")
    # print(f"wrist joint state shape: {wrist_js.shape}")
    # print(f"shaft joint state shape: {shaft_js.shape}")