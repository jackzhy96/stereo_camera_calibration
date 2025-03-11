import os
import sys
import json
import numpy as np
from typing import Tuple, List
from glob import glob
dynamic_path = os.path.abspath(__file__+"/../")
# print(dynamic_path)
sys.path.append(dynamic_path)

def load_param(file_name:str)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load the stereo calibration parameters from a json file
    :param file_name: the json file name
    :return: 8 x ndarray with stereo calibration parameters
    '''

    with open(file_name, "r") as f:
        data = json.load(f)
    cameraMatrixL = np.array(data["cameraMatrixL"], dtype=np.float32)
    distCoeffsL = np.array(data["distCoeffsL"], dtype=np.float32)
    cameraMatrixR = np.array(data["cameraMatrixR"], dtype=np.float32)
    distCoeffsR = np.array(data["distCoeffsR"], dtype=np.float32)
    R = np.array(data["R"], dtype=np.float32)
    T = np.array(data["T"], dtype=np.float32)
    E = np.array(data["E"], dtype=np.float32)
    F = np.array(data["F"], dtype=np.float32)

    return cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F


def load_file_calib(data_folder:str)->Tuple[List[str], List[str]]:
    left_img_path = sorted(glob(os.path.join(data_folder, "left_*.png")),
                           key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]))
    right_img_path = sorted(glob(os.path.join(data_folder, "right_*.png")),
                           key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]))
    return left_img_path, right_img_path

if __name__ == "__main__":
    test = 1