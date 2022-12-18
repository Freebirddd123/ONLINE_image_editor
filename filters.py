import cv2
import numpy as np
import streamlit as st


def bw_filter(img):
    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    return img_gray

def vignette(img, level=2):
    height, width = img.shape[:2]

    # Generate vignette mask using Gaussian kernels.
    X_resultant_kernel = cv2.getGaussianKernel(width, width / level)
    Y_resultant_kernel = cv2.getGaussianKernel(height, height / level)

    # Generating resultant_kernel matrix.
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()

    img_vignette = np.copy(img)

    # Apply the mask to each channel in the input image.
    for i in range(3):
        img_vignette[:, :, i] = img_vignette[:, :, i] * mask

    return img_vignette

def sepia(img):
        img_sepia = img.copy()
        # Converting to RGB as sepia matrix below is for RGB.
        img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
        img_sepia = np.array(img_sepia, dtype=np.float64)
        img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                        [0.349, 0.686, 0.168],
                                                        [0.272, 0.534, 0.131]]))
        # Clip values to the range [0, 255].
        img_sepia = np.clip(img_sepia, 0, 255)
        img_sepia = np.array(img_sepia, dtype=np.uint8)
        img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_RGB2BGR)
        return img_sepia

def pencil_sketch(img, ksize=5):
    img_blur = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
    img_sketch, _ = cv2.pencilSketch(img_blur)
    return img_sketch

def brightness(img,intensity=0):
    x=intensity
    if intensity<0:
        x=-1*intensity
    matrix = np.ones(img.shape, dtype='uint8') * x
    if intensity<0:
        return cv2.subtract(img, matrix)
    img_brighter = cv2.add(img, matrix)
    return img_brighter
def contrast(img,percent=100):
    x=percent/100
    matrix1 = np.ones(img.shape) * x
    return np.uint8(np.clip(cv2.multiply(np.float64(img), matrix1) , 0, 255))