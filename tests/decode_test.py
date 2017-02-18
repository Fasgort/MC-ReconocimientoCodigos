# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 15:23:10 2017

@author: Fasgort
"""

import cv2

img_gray = cv2.imread('test2.png', 0)
y_center = int(img_gray.shape[0]/2)
scanline = img_gray[y_center-1:y_center,0:img_gray.shape[1]]
print(scanline.shape)
for p in range(scanline.shape[1]-1):
    print(scanline.item(0,p))
cv2.imwrite('scanline.png',scanline)
    