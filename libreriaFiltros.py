import math
import cv2
import numpy as np
import sys

def convertirEscalaGrisesNTSC(imagen):
    largo, ancho, canales = imagen.shape

    imgEscalaGrises = np.zeros((largo, ancho, 1),np.uint8)
    
    for i in range(largo):
        for j in range(ancho):
            pixel = imagen[i][j]
            azul = pixel[0]
            verde = pixel[1]
            rojo = pixel[2]
            
            imgEscalaGrises[i][j] = 0.299 * azul + 0.587 * verde + 0.11 * rojo
            
    return imgEscalaGrises

def crearMatrizRelleno(imagen, mascSize):
    largo, ancho, canales = imagen.shape
    difBordes = mascSize - 1
    bordesSize = int(difBordes / 2)
    
    largoRelleno = largo + difBordes
    anchoRelleno = ancho + difBordes
    
    matrizRelleno = np.zeros((largoRelleno, anchoRelleno, 1), np.uint8)
    
    
    for i in range(bordesSize, largoRelleno - bordesSize):
        for j in range(bordesSize, anchoRelleno - bordesSize):
            matrizRelleno[j][i] = imagen[j - bordesSize][i - bordesSize]
            
    return matrizRelleno
    

def aplicarFiltro(imagen, matrizRelleno, mascara, mascSize):
    largo, ancho, canales = imagen.shape
    
    imgFiltroAplicado = np.zeros((largo, ancho, 1), np.float32)
    
    for i in range(largo):
        for j in range(ancho):
            val = (convolucionPixel(matrizRelleno, mascara, mascSize, i, j))
            imgFiltroAplicado[i][j] = val
            
    return imgFiltroAplicado
            
def convolucionPixel(matrizRelleno, mascara, mascSize, x, y):
    
    limites = int((mascSize - 1) / 2)
    sumatoriaFiltro = 0.0
    
    for i in range(-limites, limites + 1):
        for j in range(-limites, limites + 1):
            valMascara = mascara[i + limites][j + limites]
            coordY = y + j + limites
            coordX = x + i + limites

            valImagen = matrizRelleno[coordX][coordY]
            
            sumatoriaFiltro += valMascara * valImagen
    
    return sumatoriaFiltro

def mascaraGaussiana(mascSize, sigma):
    limite = int((mascSize - 1) / 2)
    gaussResultado= 0.0
    mascara = np.zeros((mascSize, mascSize), float)
    sum = 0.0
    
    s = 2.0 * sigma * sigma;
    
    for x in range(-limite, limite + 1):
        for y in range(-limite, limite + 1):
            
            r = math.sqrt(x * x + y * y);
            z = (math.exp(-(r * r) / s)) / (math.pi * s);
            gaussResultado = (math.exp(-(r * r) / s)) / (math.pi * s);
            mascara[x + limite][y + limite] = gaussResultado;
            
            sum += gaussResultado
            
    for i in range(mascSize):
        for j in range(mascSize):
            mascara[i][j] /= sum
              
    return mascara

def obtenerHistograma(imagen):
    largo, ancho, canales = imagen.shape
    
    histograma = np.zeros((256))
    nivelIntensidad = 0
    
    for i in range(largo):
        for j in range(ancho):
            nivelIntensidad = imagen[i][j]
            histograma[nivelIntensidad] += 1
            
    return histograma

def calcularPeso(histograma, inicio, final):
    total = 0
    totalAux = 0
    peso = 0

    for i in range(inicio, final + 1):

        totalAux += histograma[i]
    

    for i in range(0, len(histograma)):
    
        total += histograma[i]
    
    if (total != 0):
        peso = totalAux / total
    
    return peso

def calcularPromedio(histograma, inicio, final):
    
    sumAux = 0
    sumF = 0
    promedio = 0

    for i in range(inicio, final + 1):
    
        sumAux += histograma[i] * i
        sumF += histograma[i]
    
    if (sumF != 0) :
        promedio = sumAux / sumF
    

    return promedio

def calcularVarianza(histograma, inicio, final):
    totalAux = 0
    sumatoria = 0
    promedio = calcularPromedio(histograma, inicio, final)
    var = 0

    for i in range(inicio, final + 1):
    
        sumatoria += pow((i - promedio), 2) * histograma[i]
        totalAux += histograma[i]
    

    if (totalAux != 0) :
        var = sumatoria / totalAux
    

    return var

def umbralAlgoritmoOTSU(histograma):
    vMinima = sys.maxsize
    umbral = 0


    for t in range(256):
    
        wb = calcularPeso(histograma, 0, t)
        vb = calcularVarianza(histograma, 0, t)

        wf = calcularPeso(histograma, t + 1, 255)
        vf = calcularVarianza(histograma, t + 1, 255)

        vw = (wb * vb) + (wf * vf)

        if (vw < vMinima) :
            vMinima = vw
            umbral = t

    return umbral

def umbralizarImagen(imagen, umbral):
    
    largo, ancho, canales = imagen.shape
    
    imagenUmbralizada = np.zeros(largo * ancho, dtype=np.uint8).reshape(largo, ancho)

    for i in range(largo):
    
        for j in range(ancho):
        
            if (imagen[i][j] > umbral) :
                imagenUmbralizada[i][j] = 0
            
            else :
                imagenUmbralizada[i][j] = 255
            

    return imagenUmbralizada