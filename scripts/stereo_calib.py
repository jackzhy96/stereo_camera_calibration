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
from utils import load_file_calib

if __name__=='__main__':
    data_folder = os.path.join(repo_path, "data", "stereo_calib_images_smallboard", "stereo_calib_images")
    left_img_path, right_img_path = load_file_calib(data_folder)

    # set internal corners amount of the chess board
    board_dim = (10, 9)
    # square size in meters
    square_size = 0.006

    # find the corner points
    objp = np.zeros((board_dim[0] * board_dim[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_dim[0], 0:board_dim[1]].T.reshape(-1, 2)
    objp *= square_size

    all_objpoints = []  # 3D corner points in real world
    all_imgpointsL = []  # 2D corner points for left camera
    all_imgpointsR = []  # 2D corner points for right camera

    criteria_subpix = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,  # Increase iteration limit
        1e-6  # Decrease epsilon for more precise corner localization
    )
    #
    assert len(left_img_path) == len(right_img_path), "Unequal number of left/right images!"

    num_imgs = len(left_img_path)
    count = 0
    print('Calibration Starts....')

    # # test to find the corners
    # imgL_path = left_img_path[0]
    # imgR_path = right_img_path[0]
    # imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
    # imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)
    # if imgL is None or imgR is None:
    #     print(f"Skipping pair: {imgL_path}, {imgR_path} (not found)")
    # retL, cornersL = cv2.findChessboardCorners(imgL, board_dim, None)
    # retR, cornersR = cv2.findChessboardCorners(imgR, board_dim, None)
    # if retL and retR:
    #     cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria_subpix)
    #     cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria_subpix)
    #
    #     objpoints.append(objp)
    #     imgpointsL.append(cornersL)
    #     imgpointsR.append(cornersR)

    for imgL_path, imgR_path in zip(left_img_path, right_img_path):
        imgL_raw = cv2.imread(imgL_path)
        imgR_raw = cv2.imread(imgR_path)

        imgL = cv2.cvtColor(imgL_raw, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(imgR_raw, cv2.COLOR_BGR2GRAY)

        # gray_left = cv2.cvtColor(imgL_raw, cv2.COLOR_BGR2GRAY)
        # gray_right = cv2.cvtColor(imgR_raw, cv2.COLOR_BGR2GRAY)

        # imgL = cv2.adaptiveThreshold(gray_left, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                       cv2.THRESH_BINARY, 11, 2)
        # imgR = cv2.adaptiveThreshold(gray_right, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                              cv2.THRESH_BINARY, 11, 2)

        if imgL is None or imgR is None:
            print(f"Skipping pair: {imgL_path}, {imgR_path} (not found)")
            continue

        # retL, cornersL = cv2.findChessboardCorners(imgL, board_dim, None)
        # retR, cornersR = cv2.findChessboardCorners(imgR, board_dim, None)

        retL, cornersL = cv2.findChessboardCorners(imgL, board_dim, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        retR, cornersR = cv2.findChessboardCorners(imgR, board_dim, flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

        if retL and retR:
            cornersL = cv2.cornerSubPix(imgL, cornersL, (11,11), (-1,-1), criteria_subpix)
            cornersR = cv2.cornerSubPix(imgR, cornersR, (11,11), (-1,-1), criteria_subpix)

            all_objpoints.append(objp)
            all_imgpointsL.append(cornersL)
            all_imgpointsR.append(cornersR)

        #     cv2.drawChessboardCorners(imgL_raw, board_dim, cornersL, retL)
        #     cv2.drawChessboardCorners(imgR_raw, board_dim, cornersR, retR)
        #     cv2.imshow('Left Corners', imgL_raw)
        #     cv2.imshow('Right Corners', imgR_raw)
        #     cv2.waitKey(500)
        #
        # cv2.destroyAllWindows()
        count += 1
        sys.stdout.write(f'\r-- Progress {count}/{num_imgs}')
        sys.stdout.flush()

    # single camera calibration
    if not all_objpoints:
        raise ValueError("No valid checkerboard corners found!")
    img_shape = cv2.imread(left_img_path[0], cv2.IMREAD_GRAYSCALE).shape[::-1]

    # Left camera calibration
    retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(
        all_objpoints, all_imgpointsL, img_shape, None, None,
    )
    retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(
        all_objpoints, all_imgpointsR, img_shape, None, None,
    )
    SINGLE_CAM_REPROJ_ERR_THRESH = 0.5 ###### change to 1.5 if you would like some flexible limits
    # Evaluate per-image reprojection errors for single cameras
    errsL = []
    errsR = []
    for i, (rvec, tvec) in enumerate(zip(rvecsL, tvecsL)):
        proj, _ = cv2.projectPoints(all_objpoints[i], rvec, tvec, mtxL, distL)
        err = cv2.norm(all_imgpointsL[i], proj, cv2.NORM_L2) / len(proj)
        errsL.append(err)
    for i, (rvec, tvec) in enumerate(zip(rvecsR, tvecsR)):
        proj, _ = cv2.projectPoints(all_objpoints[i], rvec, tvec, mtxR, distR)
        err = cv2.norm(all_imgpointsR[i], proj, cv2.NORM_L2) / len(proj)
        errsR.append(err)

    # Reject the outliners
    keep_indices = []
    for i in range(len(all_objpoints)):
        if errsL[i] < SINGLE_CAM_REPROJ_ERR_THRESH and errsR[i] < SINGLE_CAM_REPROJ_ERR_THRESH:
            keep_indices.append(i)

    if len(keep_indices) < len(all_objpoints):
        print(f"\n Rejecting {len(all_objpoints) - len(keep_indices)} outlier frames based on reprojection error.")

    filtered_objpoints = [all_objpoints[i] for i in keep_indices]
    filtered_imgpointsL = [all_imgpointsL[i] for i in keep_indices]
    filtered_imgpointsR = [all_imgpointsR[i] for i in keep_indices]

    # Recalibrate single cameras with filtered images
    retL2, mtxL2, distL2, rvecsL2, tvecsL2 = cv2.calibrateCamera(
        filtered_objpoints, filtered_imgpointsL, img_shape, None, None,
    )
    retR2, mtxR2, distR2, rvecsR2, tvecsR2 = cv2.calibrateCamera(
        filtered_objpoints, filtered_imgpointsR, img_shape, None, None,
    )

    print("\nLeft single-cam RMS after outlier removal:", retL2)
    print("Right single-cam RMS after outlier removal:", retR2)

    # stereo calibration
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7)
    flags = cv2.CALIB_FIX_INTRINSIC
    retStereo, camMatL, distCoefL, camMatR, distCoefR, R, T, E, F = cv2.stereoCalibrate(
        filtered_objpoints,
        filtered_imgpointsL,
        filtered_imgpointsR,
        mtxL2,
        distL2,
        mtxR2,
        distR2,
        img_shape,
        criteria=criteria_stereo,
        flags=flags
    )

    print("Stereo Calibration RMS error:", retStereo)

    # save to json
    calib_data = {
        "cameraMatrixL": camMatL.tolist(),
        "distCoeffsL": distCoefL.tolist(),
        "cameraMatrixR": camMatR.tolist(),
        "distCoeffsR": distCoefR.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist()
    }

    with open("stereo_calib_params.json", "w") as f:
        json.dump(calib_data, f, indent=4)

    print("Stereo calibration parameters saved to 'stereo_calib_params.json'")

    baseline = np.linalg.norm(T)
    print(f'Baseline length: {baseline:.4f} meters')
