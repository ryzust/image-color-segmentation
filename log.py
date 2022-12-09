import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, convolve2d
from scipy.ndimage import filters

def LoGMask(mascSize, sigma):
    amountSlide = int((mascSize - 1) / 2)
    logResultado = 0.0
    mask = np.zeros((mascSize, mascSize), float)

    for x in range(-amountSlide, amountSlide + 1):
        for y in range(-amountSlide, amountSlide + 1):
            a = 1 / (2 * (3.1416) * sigma**4)
            b = 2 - ((x**2 + y**2) / sigma**2)
            c = - ((x**2 + y**2) / (2 * sigma**2))
            d = math.exp(c)

            logResultado = a * b * d

            mask[x + amountSlide][y + amountSlide] = logResultado

    return mask

def zero_crossing(img,delta):
    w,h= img.shape
    res = np.zeros((w,h,1),dtype=np.uint8)
    for x in range(1,w-1):
        for y in range(1,h-1):
            window = img[x-1:x+2,y-1:y+2]
            wmax = window.max()
            wmin = window.min()
            zeroCross = False
            # there are 4 cases of zero-crossing, but all can be reduced to:
            if wmax > 0 and wmin < 0:
                zeroCross = True
            
            # if a zero-cross condition and the delta are met
            if zeroCross and (wmax - wmin) > delta:
                res[x,y] = 255

    return res


def toGrayscale(img):
    w, h, c = img.shape

    gray = np.zeros((w, h, 1), np.uint8)

    for i in range(w):
        for j in range(h):
            b, g, r = img[i][j]

            gray[i][j] = 0.299 * b + 0.587 * g + 0.11 * r

    return gray

def algoritmoLaplacianoGauss(img,mascSize, sigma, delta):
    mascaraLog = LoGMask(mascSize, sigma)
    imgEscalaGrises = toGrayscale(img)
    imgLog = convolve2d(imgEscalaGrises[:, :, 0], mascaraLog)
    z = zero_crossing(imgLog, delta)
    return z
