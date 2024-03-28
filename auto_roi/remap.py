import cv2
import numpy as np


# frame = cv2.imread("5.png")

dots_rois = [
 [   1,    3,  102,  150,],
 [   3,  924,   93,  124,],
 [1279,  942,  159,  106],
]

# dots = [ frame[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] for r in dots_rois]


def adjust_offset(coordinates, offsets):
    return np.array(coordinates) +np.array(offsets)
    

def find_circle(image):
    gray_blurred = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray_blurred,  
                        cv2.HOUGH_GRADIENT, 1, minDist=100, param1=300, 
                    param2=11 , minRadius=10, maxRadius=50)
    
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(image,(i[0],i[1]),2,(0,0,255),3)

    return circles.reshape(1,3)[0,:2]



def correlação_planar(vetor_pontos_a, A_ref, B_ref ):

    # print(vetor_pontos_a)
    # Construindo as matrizes de entrada e saída
    A = np.vstack([np.array(A_ref).T, [1, 1, 1]])
    B = np.array(B_ref).T

    # print(B)
    # Calculando a matriz de transformação
    T = np.dot(B, np.linalg.inv(A))

    # Lista para armazenar os pontos equivalentes no Plano B
    pontos_b = []

    # Iterar sobre os pontos de entrada
    for ponto_a in vetor_pontos_a:
        # Convertendo ponto de entrada para uma matriz coluna
        ponto_a = tuple(np.array(tuple(ponto_a) + (1,)).reshape(-1, 1))
        
        # Aplicando a transformação para encontrar o equivalente no Plano B
        ponto_b = np.dot(T, ponto_a)
        
        # Adicionando as coordenadas transformadas à lista de pontos B
        pontos_b.append(np.round(ponto_b[:2].flatten(), 2))

    return np.array(pontos_b)


def get_A_ref(rois, frame):
    offset = [ (r[0], r[1]) for r in rois ]
    coordinates = [ find_circle(frame[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]) for r in rois ]

    return adjust_offset(coordinates, offset)

# # print(C.reshape(-1,2))
# print(correlação_planar([(20,50)], A_ref=C, B_ref=C*2))
# # print(correlação_planar(C.tolist()))

# cv2.imshow('frame', frame)
# cv2.waitKey(500)