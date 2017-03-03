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
        
        for rotated in range(4):
    
            if rotated < 2:
                inclination_corrected = simple_rotate(img, rotated)
            elif rotated == 2:
                _, mask = color_filter(img)                  
                inclination_corrected = rotate(img, mask)
            else:
                inclination_corrected = img 
            
            # Obtener máscara de filtro de color
            color_filtered, mask = color_filter(inclination_corrected)
        
            # Cambiar a espacio de color escala de grises
            gray = cv2.cvtColor(inclination_corrected, cv2.COLOR_BGR2GRAY)
        
            # Detección de bordes
            edges = edge_detection(gray)
    
            # Probar para 4 posibles distancias
            for size_correction in range(5):
                barcode_data = 'Exception'
                
                # Extraer componentes conexas y envolvente del CdB
                connected_components_detected = connected_components(edges, mask, size_correction)
                barcode, barcode_selected = barcode_detection(connected_components_detected, images[image_name])
        
                for bar_rotate in range(3):    
    
                    # Mejorar CdB
                    barcode = rotate_bar(barcode, bar_rotate)
                    barcode_processed = barcode_postprocess(barcode)
        
                    # Probar escaneado a 8 alturas diferentes
                    for scanline in range(1, 8):
                        try:
                            barcode_binarized = barcode_extractor(barcode_processed, scanline/2)
                        except IndexError:
                            barcode_data = 'Exception'
                            continue
        
                        # Algoritmo decodificación
                        barcode_data = barcode_decode(barcode_binarized)
                        if 'E' not in barcode_data:
                            #print("Resultado adquirido con parámetros: rotated("
                            #+str(rotated)+"); size_correction("
                            #+str(size_correction)+"); bar_rotate("
                            #+str(bar_rotate)+"); scanline("+str(scanline)+").")
                            break
                        #else:
                            #print("=> {}\t\t{}".format(image_name, barcode_data)) # Depuración de errores

                    if 'E' not in barcode_data:
                        break
                    #else:
                        #cv2.imwrite(image_name+str(rotated)+str(size_correction)+str(bar_rotate)+image_name,barcode_processed)
                
                if 'E' not in barcode_data:
                    break
                
            if 'E' not in barcode_data:
                break
            
        if 'E' not in barcode_data:
            print("=> {}\t\t{}".format(image_name, barcode_data))
            print()
        else:
            print("=> {}\t\t{}".format(image_name, "ERROR"))
            print()


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
