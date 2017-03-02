#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os

from functions import *

__date__ = '2017.02.15'

"""
MC - D2 - Barcode detection & decoding
"""
_DEFAULT_OUTPUT_PATH = './Output/'
_DEFAULT_INPUT_PATH = './Resources/'


def main(args):
    # Load images
    images = load_images(args.input)

    for image_name, img in images.items():
        # Obtener máscara de filtro de color
        color_filtered, mask = color_filter(img)
    
        # Cambiar a espacio de color escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Rotar imagen
        # inclination_corrected = inclination_correction(gray)
    
        # Detección de bordes
        # edges = edge_detection(inclination_corrected)
        edges = edge_detection(gray)

        # Probar para 4 posibles distancias
        for size_correction in range(5):
            barcode_data = 'E'

            # Extraer componentes conexas y envolvente del CdB
            connected_components_detected = connected_components(edges, mask, size_correction)
            barcode, barcode_selected = barcode_detection(connected_components_detected, images[image_name])

            # Mejorar CdB
            barcode_processed = barcode_postprocess(barcode)
            # Si está en vertical no procesar
            if barcode_processed.shape[1] < barcode_processed.shape[0]/3:
                break

            # Probar escaneado a 8 alturas diferentes
            for i in range(1, 8):
                try:
                    barcode_binarized = barcode_extractor(barcode_processed, i/2)
                except IndexError:
                    break

                # Algoritmo decodificación
                barcode_data = barcode_decode(barcode_binarized)
                if 'E' not in barcode_data:
                    break
            # Imprime resultado
            print("=> {}\t\t{}".format(image_name, barcode_data))
            # Si éxito salir
            if 'E' not in barcode_data:
                break


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
