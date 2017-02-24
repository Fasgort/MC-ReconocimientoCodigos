#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functions import *

__date__ = '2017.02.23'


def put_text(image, text, pos=1):
    row = 25 * pos
    cv2.putText(image, text, (20, 20 + row), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))


def nothing(*arg):
    pass


def trackbar(window_name, trackbar_name, lower, upper):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(trackbar_name, window_name, lower, upper, nothing)
    return trackbar_name


def resize(img):
    # Resize
    height, width = img.shape[:2]
    scale_ratio = 600.0 / width
    resize = cv2.resize(img, (int(scale_ratio * width), int(scale_ratio * height)),
                        interpolation=cv2.INTER_CUBIC)
    return resize
