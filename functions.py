import cv2
from PIL import Image


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
    # Eliminar ruido
    smooth = cv2.GaussianBlur(image, (3, 3), 0)
    # Aplicar gradiente en ambos ejes
    gradX = cv2.Sobel(smooth, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(smooth, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # Restar gradientes para obtener bordes en ambos ejes
    gradient = cv2.subtract(gradX, gradY)
    # Obtener valores absolutos para obtener imagen válida
    res = cv2.convertScaleAbs(gradient)
    return res


def barcode_enhance(edges):
    pass


def barcode_detection(barcode_enhanced):
    pass


def barcode_decode(barcode_selected):
    pass
