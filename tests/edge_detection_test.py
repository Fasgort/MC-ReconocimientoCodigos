#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import unittest
from main import load_images, _DEFAULT_INPUT_PATH, _DEFAULT_OUTPUT_PATH

from functions import *

__date__ = '2017.02.16'


class EdgeDetectionTestCase(unittest.TestCase):
    def setUp(self):
        self.output_path = "../" + _DEFAULT_OUTPUT_PATH
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.images = load_images("../" + _DEFAULT_INPUT_PATH)

    def test_edge_1(self):
        gray = cv2.cvtColor(self.images['test1.jpg'], cv2.COLOR_BGR2GRAY)
        edges = edge_detection(gray)

        #cv2.imshow(self.test_edge_1.__name__, edges)
        #cv2.waitKey(0);
        cv2.imwrite(self.output_path + self.test_edge_1.__name__ + '.jpg', edges)


if __name__ == '__main__':
    unittest.main()
