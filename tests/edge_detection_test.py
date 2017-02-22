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
        gray = cv2.cvtColor(self.images['test3.jpg'], cv2.COLOR_BGR2GRAY)
        edges = edge_detection(gray)

        cv2.imshow(self.test_edge_1.__name__, edges)
        cv2.waitKey(0);
        # cv2.imwrite(self.output_path + self.test_edge_1.__name__ + '.jpg', edges)

    def test_connected_components_1(self):
        gray = cv2.cvtColor(self.images['test3.jpg'], cv2.COLOR_BGR2GRAY)
        edges = edge_detection(gray)
        connected_components_detected = connected_components(edges)

        cv2.imshow(self.test_connected_components_1.__name__, connected_components_detected)
        cv2.waitKey(0);
        # cv2.imwrite(self.output_path + self.test_connected_components_1.__name__ + '.jpg', connected_components_detected)

    def test_connected_components_n(self):
        for img_name, original in self.images.items():
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            edges = edge_detection(gray)
            connected_components_detected = connected_components(edges)

            cv2.imshow(self.test_connected_components_1.__name__ + img_name, connected_components_detected)
            cv2.waitKey(0);
            # cv2.imwrite(self.output_path + self.test_connected_components_1.__name__ + '.jpg', connected_components_detected)

    def test_barcode_detection_1(self):
        img_name = 'test6.jpg'
        original = self.images[img_name]
        gray = cv2.cvtColor(self.images[img_name], cv2.COLOR_BGR2GRAY)
        edges = edge_detection(gray)
        connected_component_detected = connected_components(edges)
        barcode, barcode_detected = barcode_detection(connected_component_detected, original)

        cv2.imshow(self.test_barcode_detection_1.__name__ + 'detected', barcode_detected)
        cv2.imshow(self.test_barcode_detection_1.__name__, barcode)
        cv2.waitKey(0);
        cv2.imwrite(self.output_path + self.test_barcode_detection_1.__name__ + '.jpg', barcode)

    def test_barcode_detection_n(self):
        for img_name, original in self.images.items():
            # Resize
            height, width = original.shape[:2]
            scale_ratio = 600.0 / width
            resized = cv2.resize(original, (int(scale_ratio * width), int(scale_ratio * height)),
                                 interpolation=cv2.INTER_CUBIC)

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            edges = edge_detection(gray)
            connected_components_detected = connected_components(edges)
            barcode, barcode_detected = barcode_detection(connected_components_detected, resized)

            cv2.imshow(self.test_barcode_detection_n.__name__ + img_name, barcode_detected)
            cv2.waitKey(0);
            # cv2.imwrite(self.output_path + self.test_barcode_detection_n.__name__+ img_name + '.jpg', barcode_detected)


    def test_barcode_extractor_1(self):
        img_name = '077652082494_6.jpg'
        original = self.images[img_name]
        # # Resize
        # height, width = original.shape[:2]
        # scale_ratio = 600.0 / width
        # resize = cv2.resize(original, (int(scale_ratio * width), int(scale_ratio * height)),
        #                      interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = edge_detection(gray)
        connected_components_detected = connected_components(edges)
        barcode, barcode_detected = barcode_detection(connected_components_detected, original)
        barcode_processed = barcode_extractor(barcode)
        cv2.imshow(self.test_barcode_extractor_1.__name__, barcode_processed)
        cv2.waitKey(0);
        # cv2.imwrite(self.output_path + self.test_barcode_enhance_1.__name__ + '.jpg', barcode_enhanced)


if __name__ == '__main__':
    unittest.main()
