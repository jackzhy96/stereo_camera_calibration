import os
import sys
import cv2
import numpy as np
from glob import glob
import json
repo_path = os.path.abspath(__file__+"/../../")
# print(dynamic_path)
sys.path.append(repo_path)
dynamic_path = os.path.abspath(__file__+"/../")
# print(dynamic_path)
sys.path.append(dynamic_path)
from utils import load_param
from utils import load_file_calib


def draw_epipolar_lines(image, num_lines=10):
    """Draw equally spaced horizontal lines on the image to check epipolar alignment."""
    height = image.shape[0]
    step = height // num_lines
    for i in range(num_lines):
        y = i * step
        cv2.line(image, (0, y), (image.shape[1], y), (255, 0, 0), 1)
    return image


if __name__=='__main__':
    data_folder = os.path.join(repo_path, "data", "stereo_calib_images_largeboard", "stereo_calib_images")
    left_img_path, right_img_path = load_file_calib(data_folder)
    param_path = os.path.join(dynamic_path, 'stereo_calib_params.json')
    cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = load_param(param_path)

    imgL_path = left_img_path[0]
    imgR_path = right_img_path[0]
    imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)
    ## test
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrixL, distCoeffsL,
        cameraMatrixR, distCoeffsR,
        imgL.shape[::-1],
        R, T,
        # alpha=0  # 0 means zoom so only valid pixels remain
    )

    # Create rectification maps for each camera
    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrixL, distCoeffsL, R1, P1, imgL.shape[::-1], cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrixR, distCoeffsR, R2, P2, imgR.shape[::-1], cv2.CV_32FC1
    )

    # Example usage to remap an image:
    rect_left = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    show_left = draw_epipolar_lines(rect_left.copy(), num_lines=12)
    show_right = draw_epipolar_lines(rect_right.copy(), num_lines=12)

    cv2.imshow("left", show_left)
    cv2.imshow("right", show_right)

    cv2.waitKey(0)
    cv2.destroyAllWindows()