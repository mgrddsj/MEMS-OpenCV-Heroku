import cv2
import streamlit as st
import numpy as np
import glob

def main():
    st.markdown("**Note:** Images are compressed for web viewing.")
    st.markdown("Display and process results may be different from original images.")
    st.markdown("However, the images are still quite large, you may need a fast Internet to open this page.")
    st.markdown("图片已针对网页浏览压缩，图片的显示和处理可能与原图有偏差。")
    st.markdown("但图片仍然较大，加载可能会比较缓慢。如果加载不出来，请确保您可以访问国际互联网。")
    st.sidebar.write("Variable control panel\n变量控制台")

    
    st.markdown("***")
    st.markdown("This is a web app that renders the images real-time, tweak the values in the variable control panel to see the changes.")
    st.markdown("For phone users, you can see the control panel by clicking the arrow on the top left of the page.")
    st.markdown("这是一个实时渲染图片网页应用，您可以更改变量控制台中的数值来查看变化。")
    st.markdown("手机用户可以通过点击页面左上方的箭头按钮来打开变量控制台。")
    st.markdown("***")

    file_list = sorted(glob.glob("img/*"))
    file_path = st.sidebar.selectbox("Image:", file_list, index=8)

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
    img = cv2.Canny(thresh, 100, 200)
    st.image(img, use_column_width=True, clamp=True, caption="Contour 轮廓")

    # 霍夫变换
    st.sidebar.write("Hough Line Transform (HLT) 霍夫变换")
    blank2 = np.zeros(img.shape, np.uint8)
    blank2 = cv2.cvtColor(blank2, cv2.COLOR_GRAY2BGR)
    houghRho = st.sidebar.slider("HLT rho (step size) 霍夫变换 rho 值（搜索步长）", min_value=1, max_value=10, value=1)
    houghThreshhold = st.sidebar.slider(
        "HLT threshold 霍夫变换阈值", min_value=1, max_value=1000, value=100)
    houghMinLineLength = st.sidebar.slider(
        "HLT min. length 霍夫最短线段长度", min_value=1, max_value=500, value=10)
    houghMaxLineGap = st.sidebar.slider("HLT max line gap 霍夫最长间隙", min_value=1, max_value=200, value=100)
    lines = cv2.HoughLinesP(img, houghRho, np.pi/180, houghThreshhold,
                            minLineLength=houghMinLineLength, maxLineGap=houghMaxLineGap)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(blank2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    st.image(blank2, use_column_width=True, clamp=True, caption="Hough line transform (line detection) 霍夫变换（直线检测）")
    st.text("lines detected: {}".format(len(lines)))

    # Harris 角点检测
    st.write("**Harris Corner Detection 角点检测**")
    with st.echo():
        corners = cv2.cornerHarris(img, blockSize=5, ksize=5, k=0.04)
        harris_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        harris_img[corners > 0.1*corners.max()] = [0, 255, 0]
        st.image(harris_img, use_column_width=True, clamp=True, caption="Harris Corner detection 角点检测")
        st.text("Corners detected: {}".format(len(corners)))
    st.write("The detected corners are drawn in green, you may need to zoom in to see them.")
    st.write("检测到的角点已用绿色像素标出，您可能需要放大来看见它们。")

    # Shi-Tomasi 角点检测
    st.write("**Shi-Tomasi Corner Detection角点检测**")
    with st.echo():
        corners_s = cv2.goodFeaturesToTrack(
            img, maxCorners=100, qualityLevel=0.01, minDistance=0)
        corners_s = np.int0(corners_s)
        shitomasi_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i in corners_s:
            x, y = i.ravel()
            cv2.circle(shitomasi_img, (x, y), 5, (0, 255, 0), -1)
        st.image(shitomasi_img, use_column_width=True,
                clamp=True, caption="Shi-Tomasi Corner detection 角点检测")
        st.text("Corners detected: {}".format(len(corners_s)))


if __name__ == "__main__":
    main()