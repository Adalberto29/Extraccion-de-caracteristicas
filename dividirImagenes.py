import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage.feature import greycomatrix, greycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk

# Funcion para obtener el inidice del valor maximo
def maximovalor(arr):
    maximo = 0
    for i in range(len(arr)):
        if (arr[i] > arr[maximo]):
            maximo = i
    return maximo


# Funcion para obtener el indice del valor minimo
def minimovalor(arr):
    minimo = 0
    for i in range(len(arr)):
        if (arr[i] < arr[minimo]):
            minimo = i
    return minimo


# Funcion para dibujar un rectangulo o circulo dependiendo del tipo que se manda, se dibuja con centro x,y
def dibujar(x, y, imagen, tipo):
    if tipo == 0:
        cv2.circle(imagen, (x, y), 10, (0, 255, 0), -1, cv2.LINE_AA)
    elif tipo == 1:
        cv2.circle(imagen, (x, y), 10, (255, 255, 0), -1, cv2.LINE_AA)
    elif tipo == 2:
        cv2.circle(imagen, (x, y), 10, (255, 0, 255), -1, cv2.LINE_AA)


# Variable parms de la funcion de evento de mouse:
# 0 el tamaño por recuadro en x
# 1 el tamaño por recuadro en y
# 2 la imagen
# 3 el titulo de la imagen
# 4 la matriz en donde se guarda la clasificacion de los segmentos
def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("x : " + str(x))
        print("y : " + str(y))
        print("Clasificacion: ")
        # Coordenadas del segemento de acuerdo a la imagen
        pt = (int(x / params[0]), int(y / params[1]))
        print("x = " + str(pt[0]) + ", y = " + str(pt[1]))
        # coordenas de la esquina superior izquierda del segmento en la imagen orginal
        esquinaSI = ((pt[0] * params[0]), (pt[1] * params[1]))
        centro = (int(params[0] / 2), int(params[1] / 2))
        # Se recupera la clasificacion de la imagen
        cla = params[4]
        # Imagen recortada, de acuerdo al segmento en donde se hizo clic
        im_crop = imagenGrid[esquinaSI[1]:int(esquinaSI[1] + params[1]),
                  esquinaSI[0]:int(esquinaSI[0] + params[0])].copy()
        # Se muestra el segmento por separado
        cv2.imshow("Corte",
                   imagenGrid[esquinaSI[1]:int(esquinaSI[1] + params[1]), esquinaSI[0]:int(esquinaSI[0] + params[0])])
        # Se cambia la clasificacion ya que se hizo clic
        if cla[pt[0]][pt[1]] == 0:
            cla[pt[0]][pt[1]] = 1
        elif cla[pt[0]][pt[1]] == 1:
            cla[pt[0]][pt[1]] = 2
        else:
            cla[pt[0]][pt[1]] = 0
        # Se cambia la marca de acuerdo a la clasificacion actualizada
        dibujar(centro[0], centro[1], im_crop, cla[pt[0]][pt[1]])
        # Se actualiza el segmento de la imagen completa
        params[2][esquinaSI[1]:int(esquinaSI[1] + params[1]), esquinaSI[0]:int(esquinaSI[0] + params[0])] = im_crop
        # Se actualiza la imagen mostrada
        cv2.imshow(params[3], params[2])


# Funcion para inicializar las marcas de acuerdo a la matriz de clasificacion
def iniciarClas(imagen, clasificacion, grid):
    for x in range(0, len(clasificacion[0])):
        for y in range(0, len(clasificacion[0])):
            esquinaSI = ((x * grid[0]), (y * grid[1]))
            centro = (int(grid[0] / 2), int(grid[1] / 2))
            im_crop = imagenGrid[esquinaSI[1]:int(esquinaSI[1] + grid[1]),
                      esquinaSI[0]:int(esquinaSI[0] + grid[0])].copy()
            dibujar(centro[0], centro[1], im_crop, 0)
            imagen[esquinaSI[1]:int(esquinaSI[1] + grid[1]), esquinaSI[0]:int(esquinaSI[0] + grid[0])] = im_crop


# Funcion que recorta la imagen original en x segementos y regresa esa lista
def recortarImagen(imagen, clasificacion, grid):
    arrayImage = [None] * (len(clasificacion))
    for i in range(len(clasificacion)):
        arrayImage[i] = [None] * len(clasificacion)
    for x in range(0, len(clasificacion[0])):
        for y in range(0, len(clasificacion[0])):
            esquinaSI = ((x * grid[0]), (y * grid[1]))
            im_crop = imagenGrid[esquinaSI[1]:int(esquinaSI[1] + grid[1]),
                      esquinaSI[0]:int(esquinaSI[0] + grid[0])].copy()
            arrayImage[x][y] = im_crop

    return arrayImage


def guardarEnArchivo(nombreArchivo, histograma):
    f = open(nombreArchivo, 'w')
    for x in histograma:
        f.write(str(x) + " ")
    f.close()


# Incio del programa principal
# Apertura de la imagen
global imagenGrid
global imagenOriginal

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Ruta de la imagen")
parser.add_argument("-d", "--fileDestino", help="Nombre del archivo de texto donde se guardara la informacion")
parser.add_argument("-g", "--grid", help="Tamaño de la segmentacion")
args = parser.parse_args()
if args.file:
    imagen = cv2.imread(args.file)
imagenOriginal = imagen
# Obtencion de propiedades de la imagen
height, width, channels = imagen.shape
# Definicion del grid, tamaño de cada recuadro
if args.grid:
    grid = int(args.grid)
GRID_X = int(width / grid)
GRID_Y = int(height / grid)
print("width = " + str(width))
print("Height = " + str(height))
print("Grid x = " + str(GRID_X))
print("Grid y = " + str(GRID_Y))
# Matriz que representa los segementos de la imagen, donde se guarda la clasificacion, se ira cambian entre 0 y 1
clasificacion = np.zeros((grid, grid))
print(clasificacion)

# Visualizacion de como quedaria la imagen dividida
for x in range(0, width - 1, GRID_X):
    cv2.line(imagen, (x, 0), (x, height), (255, 0, 0), 1, 1)
for y in range(0, height - 1, GRID_Y):
    cv2.line(imagen, (0, y), (width, y), (255, 0, 0), 1, 1)
# Se guarda esta parte como raw para cambiar la clasificacion
imagenGrid = imagen.copy()
# Se inicia la imagen como no incendio todos los segmentos
iniciarClas(imagen, clasificacion, (GRID_X, GRID_Y))

# Seleccion de region en una imagen
n = 1
titulo = "Imagen " + str(n)
cv2.namedWindow(titulo)
# Se agrega funcion para eventos de mouse
cv2.setMouseCallback(titulo, on_mouse, (GRID_X, GRID_Y, imagen, titulo, clasificacion))
cv2.imshow(titulo, imagen)

# Funcion que hace que el sistema espere a que una tecla sea presionada
cv2.waitKey(0)

# En esta variable se guardan la imagen segementada
arrayIm = recortarImagen(imagenOriginal, clasificacion, (GRID_X, GRID_Y))

contador = 1
coma = ", "
destino = ""
if args.fileDestino:
    destino = args.fileDestino
f = open(destino, 'w')

for x in range(grid):
    for y in range(grid):
        print("Segmento " + " x: " + str(x) + " y: " + str(y))
        # Se divide la imagen en los canales BGR
        split = cv2.split(arrayIm[x][y])
        for i in range(3):
            im = split[i].ravel()
            resultados = plt.hist(im, 256, [0, 256], density=True)
            plt.close('all')
            histograma = resultados[0]
            mostrar = str(format(histograma.max(), '.5f'))
            mostrar += coma + str(maximovalor(histograma))
            mostrar += coma + str(format(histograma.min(), '.5f'))
            mostrar += coma + str(minimovalor(histograma))
            mostrar += coma + str(format(np.mean(histograma), '.5f'))
            mostrar += coma + str(format(np.std(histograma), '.5f'))
            mostrar += coma + str(format(np.median(histograma), '.5f'))
            mostrar += coma + str(format(np.var(histograma), '.5f'))
            mostrar += coma + str(int(clasificacion[x][y]))
            f.write(mostrar + "\n")

            g = greycomatrix(split[i], [1], [0, np.pi / 2], normed=True, symmetric=True)
            contrast = greycoprops(g, 'contrast')
            dissimilarity = greycoprops(g, 'dissimilarity')
            homogeneity = greycoprops(g, 'homogeneity')
            energy = greycoprops(g, 'energy')
            correlation = greycoprops(g, 'correlation')
            asm = greycoprops(g, 'ASM')
            entropia = format(np.sum(entropy(split[i], disk(5))), '.5f')

f.close()

# cv2.waitKey(0)
