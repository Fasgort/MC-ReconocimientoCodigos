#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import unittest

from main import load_images, _DEFAULT_INPUT_PATH, _DEFAULT_OUTPUT_PATH

from functions import *
from tests import utils

__date__ = '2017.02.16'


class EdgeDetectionTestCase(unittest.TestCase):
    def setUp(self):
        # self.image_name = '739455002382_1.jpg'
        self.image_name = 'g1.jpg'#'20170223_151830noise.jpg'  # 'test2.jpg'
        self.output_path = "../" + _DEFAULT_OUTPUT_PATH
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.images = load_images("../" + _DEFAULT_INPUT_PATH)

    def test_color_filter_1(self):
        original = self.images[self.image_name]
        resize = utils.resize(original)
        color_filtered, mask = color_filter(resize)

        cv2.imshow(self.test_color_filter_1.__name__, color_filtered)
        cv2.waitKey(0)
        # cv2.imwrite(self.output_path + self.test_color_filter_1.__name__ + self.image_name, color_filtered)
        # cv2.imwrite(self.output_path + self.test_color_filter_1.__name__ +'-mask-'+ self.image_name, mask)

    def test_color_filter_n(self):
        for img_name, original in self.images.items():
            resize = utils.resize(original)
            color_filtered, mask = color_filter(resize)

            cv2.imshow(self.test_edge_1.__name__ + img_name, color_filtered)
            cv2.waitKey(0)
            # cv2.imwrite(self.output_path + self.test_color_filter_n.__name__ + img_name+ '.jpg', color_filtered)

    def test_edge_1(self):
        original = self.images[self.image_name]
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = edge_detection(gray)
        resize = utils.resize(edges)
        cv2.imshow(self.test_edge_1.__name__, resize)
        cv2.waitKey(0)
        cv2.imwrite(self.output_path + self.test_edge_1.__name__ + '.jpg', edges)

    def test_edge_n(self):
        for img_name, original in self.images.items():
            resize = utils.resize(original)
            color_filtered, mask = color_filter(resize)
            gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
            edges = edge_detection(gray)

            cv2.imshow(self.test_edge_1.__name__ + img_name, edges)
            cv2.imshow(self.test_edge_1.__name__ + 'no_color', cv2.bitwise_and(edges, edges, mask=mask))
            cv2.waitKey(0)
            # cv2.imwrite(self.output_path + self.test_edge_1.__name__ + '.jpg', edges)

    def test_connected_components_adaptive(self):
        original = self.images[self.image_name]
        for image_name, original in self.images.items():
            height, width = original.shape[:2]
            factor = (width / 1600) *0.5
            print("{};{};".format(image_name, width))

        # color_filtered, mask = color_filter(original)
        # gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        # edges = edge_detection(gray)
        # connected_components_detected = connected_components(edges, mask, factor)
        #
        # resize = utils.resize(connected_components_detected)
        # cv2.imshow(self.test_connected_components_1.__name__, resize)
        # cv2.waitKey(0)
        '''
        '''

    def test_connected_components_1(self):
        original = self.images[self.image_name]
        color_filtered, mask = color_filter(original)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = edge_detection(gray)
        connected_components_detected = connected_components(edges, mask,1)

        resize = utils.resize(connected_components_detected)
        cv2.imshow(self.test_connected_components_1.__name__, resize)
        cv2.waitKey(0)
        # cv2.imwrite(self.output_path + self.test_connected_components_1.__name__ + '.jpg', connected_components_detected)
        # cv2.imwrite(self.output_path + self.test_connected_components_1.__name__ + '1.jpg', edges)
        # cv2.imwrite(self.output_path + self.test_connected_components_1.__name__ + '2.jpg', closed)
        # cv2.imwrite(self.output_path + self.test_connected_components_1.__name__ + '3.jpg', eroded)
        # cv2.imwrite(self.output_path + self.test_connected_components_1.__name__ + '4.jpg', dilated)

    def test_connected_components_n(self):
        for img_name, original in self.images.items():
            # if img_name[:13] == '20170223_1517':
            color_filtered, mask = color_filter(original)
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            edges = edge_detection(gray)
            connected_components_detected = connected_components(edges, mask )
            resize = utils.resize(connected_components_detected)
            cv2.imshow(self.test_connected_components_n.__name__ + img_name, resize)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def test_barcode_detection_1(self):
        original = self.images[self.image_name]
        color_filtered, mask = color_filter(original)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = edge_detection(gray)
        for i in range(3):
            connected_component_detected = connected_components(edges,mask,i)
            barcode, barcode_detected = barcode_detection(connected_component_detected, original)

            cv2.imshow(self.test_barcode_detection_1.__name__ + 'detected', utils.resize(barcode_detected))
            cv2.imshow(self.test_barcode_detection_1.__name__, utils.resize(barcode))
            cv2.waitKey(0)
        # cv2.imwrite(self.output_path + self.test_barcode_detection_1.__name__ + '.jpg', barcode)
        # cv2.imwrite(self.output_path + self.test_barcode_detection_1.__name__ + 'barcode_detected.jpg', barcode_detected)

    def test_barcode_detection_n(self):
        for img_name, original in self.images.items():
            #if img_name[:9] == '20170223_':
                color_filtered, mask = color_filter(original)
                gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                edges = edge_detection(gray)
                connected_components_detected = connected_components(edges, mask,1)
                barcode, barcode_detected = barcode_detection(connected_components_detected, original)
                resized = utils.resize(barcode_detected)
                # cv2.imshow(self.test_barcode_detection_n.__name__ + img_name, resized)
                # cv2.waitKey(0)
                cv2.imwrite(self.output_path + 'barcode_' + img_name + '.jpg', barcode_detected)

    def test_barcode_extractor_1(self):
        original = self.images[self.image_name]
        color_filtered, mask = color_filter(original)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = edge_detection(gray)
        connected_components_detected = connected_components(edges, mask)
        barcode, barcode_detected = barcode_detection(connected_components_detected, original)
        barcode_processed = barcode_extractor(barcode)
        print(*barcode_processed)
        cv2.imshow(self.test_barcode_extractor_1.__name__, barcode)
        cv2.waitKey(0)

    def test_barcode_extractor_n(self):
        for img_name, original in self.images.items():
            color_filtered, mask = color_filter(original)
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            edges = edge_detection(gray)
            connected_components_detected = connected_components(edges, mask)
            barcode, barcode_detected = barcode_detection(connected_components_detected, original)
            barcode_processed = barcode_extractor(barcode)
            print(*barcode_processed)
            cv2.imshow(self.test_barcode_extractor_n.__name__ + img_name, original)
            cv2.waitKey(0)


if __name__ == '__main__':
    unittest.main()
