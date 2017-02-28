import cv2
import numpy as np


def inclination_correction(image):
    """ Corrects image inclination
    Args:
        image (image) Image to rotate
    Returns:
        (image) Image with correct inclination
    """
    return image


def color_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_limit = 56  # 51
    s_limit = 56  # 148
    lower_white = np.array([0, 0, v_limit], dtype=np.uint8)
    upper_white = np.array([255, s_limit, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask_dilated = cv2.dilate(mask, None, iterations=2)
    return cv2.bitwise_and(image, image, mask=mask_dilated), mask


def edge_detection(image):
    """ Extracts edges from image
    Args:
        image (gray scale image) Image to rotate
    Returns:
        (image) Image with borders highlighted
    """
    # http://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    # Ref. http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html#why-laplacian-is-a-high-pass-filter
    # Eliminar ruido
    # Sal y pimienta
    blur = cv2.medianBlur(image, 3)
    # Filtro bilateral (que preserva bordes)
    smooth = cv2.bilateralFilter(blur, 11, 17, 17)
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
    (_, thresh) = cv2.threshold(smoothed_edges, 80, 255, cv2.THRESH_BINARY)
    res = thresh
    return res


def connected_components(edges, mask=None, size_correction=0):
    if size_correction == 0:
        strut_x = 46
        strut_y = 15
    elif size_correction == 1:
        strut_x = 80
        strut_y = 15
    elif size_correction == 2:
        strut_x = 150
        strut_y = 80
    else:
        strut_x = 46
        strut_y = 15
    if mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    # Operaciones morfológicas: closing & erosion
    # Ref. http://docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html
    closing_mask = cv2.getStructuringElement(cv2.MORPH_RECT, (strut_x, strut_y))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, closing_mask)
    # Ref. http://docs.opencv.org/2.4/doc/tutorials/imgproc/erosion_dilatation/erosion_dilatation.html
    eroded = cv2.erode(closed, None, iterations=4)
    dilated = cv2.dilate(eroded, None, iterations=4)
    # cv2.imshow('erode', eroded)
    # cv2.imshow('dilate', dilated)
    # Detectar componentes conectados
    # Ref. http://aishack.in/tutorials/labelling-connected-components-example/
    """ connected:
        num_labels = connected[0]
        labels = connected[1]
        stats = connected[2]
        centroids = connected[3]
    """
    connected = cv2.connectedComponents(dilated)
    # Asignar a cada uno de los componentes un valor diferenciador
    connected_components = np.uint8((connected[1] * 255) / connected[0])
    res = connected_components
    return res


def barcode_detection(connected_component, original_img=None):
    barcode_img = None
    # Ref. http://docs.opencv.org/3.2.0/d4/d73/tutorial_py_contours_begin.html
    (_, contours, _) = cv2.findContours(connected_component.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filtro solo el mayor
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    # Definir límites de contorno
    # Ref. http://opencvpython.blogspot.com.es/2012/06/contours-2-brotherhood.html
    c_start_x, c_start_y, c_wide, c_height = cv2.boundingRect(c)
    # Recortar solo código de barras
    if original_img is not None:
        barcode_img = np.copy(original_img[c_start_y:c_start_y + c_height, c_start_x:c_start_x + c_wide])
    # Dibujar contorno
    # Ref. http://docs.opencv.org/2.4.2/modules/core/doc/drawing_functions.html#drawcontours
    barcode_highlighted = original_img.copy()
    barcode_highlighted = cv2.rectangle(barcode_highlighted, (c_start_x, c_start_y),
                                        (c_start_x + c_wide, c_start_y + c_height), (0, 0, 255), 2)
    res = barcode_img
    return res, barcode_highlighted


def barcode_extractor(barcode_img, scan_x_pos=2):
    """ Extract barcode scan data
    Args:
        barcode_img (BGR image) Barcode image
        scan_x_pos (int) Height scan position in (0,4)
    Returns:
        (list) Data coded in binary list
    """

    if barcode_img.shape[2] > 1:
        gray = cv2.cvtColor(barcode_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = barcode_img
    img = gray
    scan_x = int((img.shape[0] * scan_x_pos) / 4)
    # 1 Compute luminance
    luminance = list()
    for x in range(img.shape[1]):
        px = img[scan_x, x]
        luminance.append(px / 255.0)
    # 2 Get locals max & min luminance
    local_max = list()
    local_min = list()
    aux = 0
    prev_state = 'dark'
    for x in range(1, len(img[scan_x]) - 1):
        # Check all pixels have luminance |max-lm(px)| - |lm(px)-min| > 0.01
        if luminance[x - 1] - luminance[x] <= 0:
            # lighter region
            if luminance[x] - luminance[x + 1] > 0:
                # darker
                if abs(aux - luminance[x]) >= 0.01:
                    if prev_state == 'dark':
                        local_max.append((x, luminance[x]))
                        prev_state = 'light'
                        aux = luminance[x]
        else:
            # darker region
            if luminance[x] - luminance[x + 1] < 0:
                # lighter
                if abs(aux - luminance[x]) >= 0.01:
                    if prev_state == 'light':
                        local_min.append((x, luminance[x]))
                        prev_state = 'dark'
                        aux = luminance[x]
    # 3 Prune dark max & light min
    local_max_mean = np.mean([lum_tup[1] for lum_tup in local_max])
    local_min_mean = np.mean([lum_tup[1] for lum_tup in local_min])
    local_prune_limit = ((local_max_mean - local_min_mean) / 2) + local_min_mean
    local_max_pruned = filter(lambda lum_tup: lum_tup[1] > local_prune_limit, local_max)
    local_min_pruned = filter(lambda lum_tup: lum_tup[1] < local_prune_limit, local_min)

    local_max_min = list()
    local_max_min.extend(local_max_pruned)
    local_max_min.extend(local_min_pruned)
    local_max_min = sorted(local_max_min, key=lambda x: x[0])

    # 4 Adaptive threshold
    res = list()
    sensitivity = int(len(img[scan_x]) / 76)
    for x in range(len(img[scan_x])):
        prev_lum = [lum_tup[1] for lum_tup in local_max_min if lum_tup[1] <= x]
        if len(prev_lum) >= sensitivity:
            prev_lum = prev_lum[-sensitivity:]
        else:
            prev_lum = [local_max_min[0][1], local_max_min[1][1]]
        lum_mean = np.mean(prev_lum)
        if lum_mean > 1 - (1 / 4):
            lum_mean = 1 - (1 / 4)
        threshold_level = lum_mean * 255
        res.append(1 if img[scan_x][x] > threshold_level else 0)
    return res


def barcode_decode(scanline_array):
    char_array = np.zeros((60, 1), int)  # Carácteres codificados en EAN-13 barcode
    change_count = 0  # Cambios entre blanco y negro en un barcode EAN-13
    last_pixel = 1  # 0: Black; 1: White
    last_decoded = 0  # Pixel donde empieza a leer

    try:
        while change_count != 60:
            pixel_count = 0
            while last_pixel is scanline_array[last_decoded]:
                last_decoded += 1
                pixel_count += 1
            char_array[change_count] = pixel_count
            last_pixel = scanline_array[last_decoded]
            change_count += 1
    except Exception:
        return "E1"  # Error E1: Scanline has not a barcode

    ean13_char_array = np.zeros((12, 4), int)
    for c in range(12):
        if c < 6:  # Blanco, negro, blanco, negro
            for v in range(4):
                ean13_char_array[c][v] = char_array[4 + c * 4 + v]
        else:  # Negro, blanco, negro, blanco
            for v in range(4):
                ean13_char_array[c][v] = char_array[9 + c * 4 + v]

    # Existen varias aproximaciones al problema de tomar el valor del carácter
    # Media de valores, mediana, etc. En este caso tomaremos el valor del pixel medio
    ean13_binarycode_array = np.zeros((12, 7), int)
    for c in range(len(ean13_char_array)):
        for v in range(7):
            length_arr = ean13_char_array[c][0] + ean13_char_array[c][1] + ean13_char_array[c][2] + ean13_char_array[c][
                3]
            pixel_pos = int((length_arr / 7) * (v + 0.5)) - ean13_char_array[c][0]
            character_pos = 0
            while pixel_pos > 0:
                character_pos += 1
                pixel_pos -= ean13_char_array[c][character_pos]
            if c < 6:  # Blanco, negro, blanco, negro
                if character_pos % 2 is 1:  # Negro
                    ean13_binarycode_array[c][v] = 1
                else:  # Blanco
                    ean13_binarycode_array[c][v] = 0
            else:  # Negro, blanco, negro, blanco
                if character_pos % 2 is 1:  # Blanco
                    ean13_binarycode_array[c][v] = 0
                else:  # Negro
                    ean13_binarycode_array[c][v] = 1

    # Etapa final, decodificación
    parity = []
    decoded_1 = []
    decoded_2 = []

    # Edge cases
    count = 0
    for v in ean13_binarycode_array[0]:
        count += v
    if count % 2 == 0:
        # gg wp - El código de barras está siendo leído al revés
        reversed_ean13_binarycode_array = np.zeros((12, 7), int)
        for c in range(12):
            for v in range(7):
                reversed_ean13_binarycode_array[c][v] = ean13_binarycode_array[11 - c][6 - v]
        ean13_binarycode_array = reversed_ean13_binarycode_array

    for c in range(6):
        count = 0
        character = []
        for v in ean13_binarycode_array[c]:
            count += v
            character.append(str(v))
        if count % 2 == 0:
            parity.append("E")
            decode_dict = {
                "0100111": "0",
                "0110011": "1",
                "0011011": "2",
                "0100001": "3",
                "0011101": "4",
                "0111001": "5",
                "0000101": "6",
                "0010001": "7",
                "0001001": "8",
                "0010111": "9",
            }
            decoded_1.append(decode_dict.get("".join(character), "E"))
        else:
            parity.append("O")
            decode_dict = {
                "0001101": "0",
                "0011001": "1",
                "0010011": "2",
                "0111101": "3",
                "0100011": "4",
                "0110001": "5",
                "0101111": "6",
                "0111011": "7",
                "0110111": "8",
                "0001011": "9",
            }
            decoded_1.append(decode_dict.get("".join(character), "E"))

    for c in range(6, 12):
        count = 0
        character = []
        for v in ean13_binarycode_array[c]:
            count += v
            character.append(str(v))
        decode_dict = {
            "1110010": "0",
            "1100110": "1",
            "1101100": "2",
            "1000010": "3",
            "1011100": "4",
            "1001110": "5",
            "1010000": "6",
            "1000100": "7",
            "1001000": "8",
            "1110100": "9",
        }
        decoded_2.append(decode_dict.get("".join(character), "E"))

    decode_dict = {
        "OOOOOO": "0",
        "OOEOEE": "1",
        "OOEEOE": "2",
        "OOEEEO": "3",
        "OEOOEE": "4",
        "OEEOOE": "5",
        "OEEEOO": "6",
        "OEOEOE": "7",
        "OEOEEO": "8",
        "OEEOEO": "9",
    }
    decoded_string = decode_dict.get("".join(parity), "E")
    decoded_string += " " + "".join(decoded_1) + " " + "".join(decoded_2)

    if 'E' in decoded_string:
        return 'E2'  # Error E2: Some characters couldn't be decoded

    return decoded_string
