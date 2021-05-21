import cv2
import streamlit as st
import numpy as np
import glob
import multiprocessing
import time
import stqdm
import os
from functools import partial

def processImages(file_path, CHESSBOARD, criteria, objpoints, imgpoints, objp):
    image = cv2.imread(file_path)  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # st.image(image, use_column_width=True, caption="原图", channels="BGR")

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        imgpoints.append(corners2)

def main():
    st.write("Images used for camera calibration 用于相机矫正的照片")
    file_list = sorted(glob.glob("camcalib/*.webp"), key=lambda x: int(os.path.basename(x).split(".")[0]))
    photo_selector = st.select_slider("", file_list, value=file_list[0])
    st.image(cv2.imread(photo_selector), channels="BGR")
    
    
    if st.button("Start camera calibration 开始相机矫正"):
        CHESSBOARD = (9, 7)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001

        manager = multiprocessing.Manager()
        # Creating vector to store vectors of 3D points for each CHESSBOARD image
        objpoints = manager.list()
        # Creating vector to store vectors of 2D points for each CHESSBOARD image
        imgpoints = manager.list()

        # Defining the world coordinates for 3D points
        objp = np.zeros((1, CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

        # Multiprocess
        start_time = time.time()
        pool = multiprocessing.Pool()
        func = partial(processImages, CHESSBOARD=CHESSBOARD, criteria=criteria, objpoints=objpoints, imgpoints=imgpoints, objp=objp)
        for _ in stqdm.stqdm(pool.imap_unordered(func, file_list), total=len(file_list), unit="photo"):
            pass
            
        pool.close()
        pool.join()
        
        st.write("Number of image used to calibrate the camera:", len(objpoints))
        st.write("Time used:", time.time()-start_time, "s")

        # 相机校准
        image = cv2.imread("camcalib/1.webp")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        st.write("Camera matrix 相机内参矩阵 mtx:")
        st.write(mtx)
        st.write("Distortion Coefficients 透镜畸变系数 dist:")
        st.write(dist)
        st.write("Rotation vectors 旋转向量 rvecs:")
        st.write(rvecs[0])
        st.write("Translation vectors 位移向量 tvecs:")
        st.write(tvecs[0])

        st.write("Using the calibration results to calibrate an image (1.webp): ")
        st.write("使用相机矫正的结果来纠正畸变（此处使用 1.webp）:")
        undistorted = cv2.undistort(image, mtx, dist)
        st.image(undistorted, use_column_width=True, caption="校正后的图像", channels="BGR")
    else:
        st.write("Press the start button to start the camera calibration process. ")
        st.write("Please **DO NOT** press it repeatedly as it may crash the server.")
        st.write("请按下按钮以开始矫正相机")
        st.write("请**不要**反复点击按钮，否则服务器可能会崩溃。")


if __name__ == "__main__":
    main()