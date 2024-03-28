import cv2
import numpy as np


# frame = cv2.imread("5.png")

# dots_rois = [
#  [   1,    3,  102,  150,],
#  [   3,  924,   93,  124,],
#  [1279,  942,  159,  106],
# ]
dots_rois = [
 [   46,    16,  70,  70,],
 [   40,  900,   70,  70,],
 [1365,  892,  70,  70],
]

# dots = [ frame[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] for r in dots_rois]


def adjust_offset(coordinates, offsets):
    return np.array(coordinates) +np.array(offsets)
    

def find_circle(image):
    image = cv2.GaussianBlur(image, (5,5), 1)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    mask = mask1+mask0
    image2 = cv2.bitwise_and(image, image, mask=mask)
    
    gray_blurred = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray_blurred,  
                        cv2.HOUGH_GRADIENT, 1, minDist=100, param1=300, 
                    param2=1.5 , minRadius=15, maxRadius=30)
    
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),1)
        # draw the center of the circle
        cv2.circle(image,(i[0],i[1]),2,(0,0,255),1)
    
        # cv2.imwrite(f"{i[2]}.jpg", image)

    return circles.reshape(1,3)[0,:2], image



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
    try:
        coordinates = []
        
        for r in rois:
            coords, fr = find_circle(frame[r[1]:r[1]+r[3], r[0]:r[0]+r[2]])
            frame[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = fr
            coordinates.append(coords)

        return np.array(coordinates), [ (r[0], r[1]) for r in rois ], frame
    except (AttributeError, TypeError):
        return None
        
if __name__ == "__main__":
    i = cv2.imread("/home/jetson/Documents/test/Transbordo/reference.png")
    img_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    mask = mask0+mask1
    cv2.imwrite('m.jpg', mask0)