#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os

from functions import *

__date__ = '2017.02.15'

"""
MC - D2 - Barcode detection & recognition
"""
_DEFAULT_OUTPUT_PATH = './Output/'
_DEFAULT_INPUT_PATH = './Resources/'


def main(args):
    # Load images
    images = load_images(args.input)
    gray = cv2.cvtColor(images['test1.jpg'], cv2.COLOR_BGR2GRAY)

    # Rotar imagen
    inclination_corrected = inclination_correction(gray)

    # Detección de bordes
    edges = edge_detection(inclination_corrected)

    # Binarizar y mejorar
    barcode_enhanced = barcode_enhance(edges)

    # Extraer componentes conectados y envolvente del CdB
    barcode_selected = barcode_detection(barcode_enhanced)

    # Algoritmo decodificación
    barcode_data = barcode_decode(barcode_selected)


def load_images(path):
    """ Returns images from given location
      Images are loaded with BGRA color space
      Args:
        path (str) Path to images.
      Returns:
        image (tuple(ndarray)) Output array filled with loaded images.
      """
    images = dict()
    if os.path.isdir(path):
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                images[filename] = img
    else:
        img = cv2.imread(path)
        if img is not None:
            filename = os.path.basename(path)
            images[filename] = img
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MC - D2 - Barcode detection & recognition")
    parser.add_argument(
        "-i",
        "--input",
        help="input data path",
        default=_DEFAULT_INPUT_PATH, type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="output data path",
        default=_DEFAULT_OUTPUT_PATH, type=str)
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # Setup logging
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    main(args)
