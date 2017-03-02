import cv2
import numpy as np
import os
import math
from main import _DEFAULT_INPUT_PATH
from functions import edge_detection
from tests import utils


def rotate(img, image_name):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = edge_detection(gray)
    img_aux = img.copy()
    # Ref. http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_aux, (x1, y1), (x2, y2), (0, 0, 255), 2)

        width, height = img_aux.shape[:2]
        if not 1.5 < theta < 1.7:  # Rotado no en [85,97]º
            # Ref. http://www.pyimagesearch.com/2014/01/20/basic-image-manipulations-in-python-and-opencv-resizing-scaling-rotating-and-cropping/
            M = cv2.getRotationMatrix2D((height / 2, width / 2), (math.degrees(theta)),
                                        0.7)
            img_aux = cv2.warpAffine(img, M, (height, width))
    return img_aux


input_path = '../' + _DEFAULT_INPUT_PATH
image_name = 'g5rot.jpg'  # 'a10r.jpg'
img = cv2.imread(os.path.join(input_path, image_name))
cv2.imshow(image_name, utils.resize(rotate(img, image_name)))
cv2.waitKey(0)
