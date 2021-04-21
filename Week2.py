import cv2
import streamlit as st
import numpy as np
from PIL import Image

file_path = "img/50x_14.webp"

st.header("")
st.markdown("**Note:** Images are compressed for web viewing.")
st.markdown("Display and process results may be different from original images.")
st.markdown("However, the images are still quite large, you may need a fast Internet to open this page.")

original_img = cv2.imread(file_path)
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
st.image(original_img, use_column_width=True, caption="Original image 原图")

image = cv2.imread(file_path, 0)  # Read as grayscale image
st.image(image, use_column_width=True, caption="Grayscale image 灰度图")

# 最大滤波处理
kernel = np.ones((3, 3), np.uint8)
dilate_iteration = st.sidebar.slider(
    "Dilate iteration 最大滤波（膨涨）次数", min_value=1, max_value=50, value=1)
dilate = cv2.dilate(image, kernel, iterations=dilate_iteration)
st.image(dilate, use_column_width=True, clamp=True, caption="Dilated image 最大滤波处理")

# 最小滤波处理
kernel = np.ones((3, 3), np.uint8)
# iteration的值越高，模糊程度(腐蚀程度)就越高 呈正相关关系且只能是整数
erosion_iteration = st.sidebar.slider(
    "Erosion iteration 最小滤波（腐蚀）次数", min_value=1, max_value=50, value=1)
erosion = cv2.erode(dilate, kernel, iterations=erosion_iteration)
st.image(erosion, use_column_width=True, clamp=True, caption="Eroded image 最小滤波处理")

# 阈值处理
threshhold_value = st.sidebar.slider(
    "Threshold 阈值", min_value=50, max_value=255, value=100)
ret, thresh = cv2.threshold(erosion, threshhold_value, 255, cv2.THRESH_BINARY)
thresh = thresh.astype(np.float64)
st.image(thresh, use_column_width=True, clamp=True, caption="Threshold processed image 阈值处理")

# 轮廓
thresh = thresh.astype(np.uint8)
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
blank = np.zeros(thresh.shape, np.uint8)
img = cv2.drawContours(blank, contours, -1, (255, 255, 255), 3)
st.image(img, use_column_width=True, clamp=True, caption="Contour 轮廓")

# 霍夫变换
blank2 = np.zeros(img.shape, np.uint8)
blank2 = cv2.cvtColor(blank2, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLinesP(img, 1, np.pi/180, 100,
                        minLineLength=100, maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(blank2, (x1, y1), (x2, y2), (0, 255, 0), 2)
st.image(blank2, use_column_width=True, clamp=True, caption="Hough line transform (line detection) 霍夫变换（直线检测）")
st.text("lines detected: {}".format(len(lines)))
