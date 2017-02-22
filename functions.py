import cv2
import numpy as np


def inclination_correction(image):
    """ Corrects image inclination
    Args:
        image (image) Image to rotate
    Returns:
        (image) Image with correct inclination
    """
    pass


def edge_detection(image):
    """ Extracts edges from image
    Args:
        image (gray scale image) Image to rotate
    Returns:
        (image) Image with borders highlighted
    """
    # http://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    # Eliminar ruido
    # Mejor con filtro bilateral (que preserva bordes)
    smooth = cv2.bilateralFilter(image, 11, 17, 17)
    # Aplicar gradiente en ambos ejes
    grad_x = cv2.Sobel(smooth, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_y = cv2.Sobel(smooth, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # Restar gradientes para obtener bordes en ambos ejes
    gradient = cv2.subtract(grad_x, grad_y)
    # Obtener valores absolutos para obtener imagen válida [0,255]
    edges = cv2.convertScaleAbs(gradient)
    # Segmentar uso umbralización para eliminar ruido de detección de bordes
    # Ref. http://docs.opencv.org/3.2.0/d7/d4d/tutorial_py_thresholding.html
    smoothed_edges = cv2.GaussianBlur(edges, (3, 3), 0)
    # Ref. http://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/threshold.html
    (_, thresh) = cv2.threshold(smoothed_edges, 200, 255, cv2.THRESH_BINARY)
    res = thresh
    return res


def connected_components(edges):
    # Operaciones morfológicas: closing & erosion
    # Ref. http://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html
    closing_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, closing_mask)
    # Ref. http://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
    eroded = cv2.erode(closed, None, iterations=4)
    dilated = cv2.dilate(eroded, None, iterations=4)
    # cv2.imshow('erode', eroded)
    # cv2.imshow('dilate', dilated)
    # Detectar componentes conectados
    # Ref. http://aishack.in/tutorials/labelling-connected-components-example/
    """ connected:
        num_labels = output[0]
        labels = output[1]
        stats = output[2]
        centroids = output[3]
    """
    connected = cv2.connectedComponents(dilated)
    # Asignar a cada uno de los componentes un valor diferenciador
    connected_components = np.uint8((connected[1] * 255) / connected[0])
    res = connected_components
    return res


def barcode_detection(edges_img, original_img=None):
    barcode_img = None
    # Ref. http://docs.opencv.org/3.2.0/d4/d73/tutorial_py_contours_begin.html
    (_, contours, _) = cv2.findContours(edges_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filtro solo el mayor
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # Definir límites de contorno
    # Ref. http://opencvpython.blogspot.com.es/2012/06/contours-2-brotherhood.html
    c_start_x, c_start_y, c_wide, c_height = cv2.boundingRect(c)
    # Dibujar contorno
    # Ref. http://docs.opencv.org/2.4.2/modules/core/doc/drawing_functions.html#drawcontours
    barcode_highlighted = cv2.rectangle(original_img, (c_start_x, c_start_y),
                                        (c_start_x + c_wide, c_start_y + c_height), (0, 0, 255), 2)

    # Recortar solo código de barras
    if original_img is not None:
        barcode_img = original_img[c_start_y:c_start_y + c_height, c_start_x:c_start_x + c_wide]
    res = barcode_img
    return res, barcode_highlighted


def barcode_enhance(barcode_img):
    pass


def barcode_decode(scanline_array):
    
    char_array = np.zeros((60, 1), int) # Carácteres codificados en EAN-13 barcode
    change_count = 0 # Cambios entre blanco y negro en un barcode EAN-13
    last_pixel = 0 # 0: White; 1: Black
    last_decoded = 0
    
    while change_count is not 60:
        pixel_count = 0
        while last_pixel is not scanline_array[last_decoded]:
            last_decoded += 1
            pixel_count += 1
        char_array[change_count] = pixel_count
        last_pixel = scanline_array[last_decoded]
        change_count += 1
    
    for c in char_array:
        print(c)
        
    ean13_char_array = np.zeros((12,4),int)
    
    for c in range(12):
        if c < 6:
            for v in range(4):
                ean13_char_array[c][v] = char_array[4+c*4+v]
        else:
            for v in range(4):
                ean13_char_array[c][v] = char_array[9+c*4+v]
    
    for c in ean13_char_array:
        print(c)
        
    ean13_binarycode_array = np.zeros((12,7), int)
    
    # Existen varias aproximaciones al problema de tomar el valor del carácter
    # Media de valores, mediana, etc. En este caso tomaremos el valor del pixel medio
    for c in ean13_char_array:
        char_arr = ean13_char_array[c][0] + ean13_char_array[c][1] 
        + ean13_char_array[c][2] + ean13_char_array[c][3]
        length_arr = len(char_arr)
        for v in ean13_char_array[c]:
            ean13_binarycode_array[c][v] = char_arr[int(((length_arr/7)*(v+0.5))/length_arr)]

    for c in ean13_binarycode_array:
        print(c)
            
    
    pass
