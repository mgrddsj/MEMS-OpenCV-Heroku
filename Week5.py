import cv2
import streamlit as st
import numpy as np
import glob
from PIL import Image

def main():
    file_path = "img/chessboard.webp"

    image = cv2.imread(file_path)  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(image, use_column_width=True, caption="Original Image 原图", channels="BGR")

    CHESSBOARD = (9, 7)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each CHESSBOARD image
    objpoints = []
    # Creating vector to store vectors of 2D points for each CHESSBOARD image
    imgpoints = [] 

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHESSBOARD[0]*CHESSBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        corners_img = cv2.drawChessboardCorners(image, CHESSBOARD, corners2,ret)
        st.image(corners_img, use_column_width=True, caption="Chessboard detection results 棋盘检测结果", channels="BGR")
        st.info("Please zoom in to see the chessboard corners found 请放大图片来看清棋盘标定结果")

if __name__ == "__main__":
    main()