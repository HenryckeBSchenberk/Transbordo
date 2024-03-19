
import numpy as np
import cv2

def calculate_average_coordinates(coordinates, threshold):
    coordinates_array = np.array(coordinates)
    
    # Calculate pairwise distances
    pairwise_distances = np.linalg.norm(coordinates_array[:, None, :] - coordinates_array, axis=-1)

    # Create a mask for coordinates that are close to each other
    close_coordinates_mask = pairwise_distances < threshold

    # Calculate the average of each group of close coordinates
    average_coordinates = np.array([np.mean(coordinates_array[mask], axis=0) for mask in close_coordinates_mask])
    average_coordinates = np.unique(average_coordinates, axis=0)
    return average_coordinates.tolist()

def rounded_rectangle(src, top_left, bottom_right, radius=1, color=(255,255,255), thickness=1, line_type=cv2.LINE_AA, from_zero=False):

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3
    if from_zero:
        bottom_right = bottom_right[0]-top_left[0], bottom_right[1]-top_left[1]
        top_left = (0,0)
    p1 = top_left
    p2 = (bottom_right[0], top_left[1])
    p3 = bottom_right
    p4 = (top_left[0], bottom_right[1])

    height = abs(bottom_right[1] - top_left[1])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (height/2))

    if thickness < 0:

        #big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
        [top_left_main_rect, bottom_right_main_rect], 
        [top_left_rect_left, bottom_right_rect_left], 
        [top_left_rect_right, bottom_right_rect_right]]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

    return src


def calculate_corners(center, width, height, as_roi=True):
    # Calculate half of width and height
    half_width = width / 2
    half_height = height / 2

    # Calculate top-left and bottom-right corners
    top_left = (int(center[0] - half_width), int(center[1] - half_height))
    if as_roi:
        return *top_left, width, height
    
    bottom_right = (int(center[0] + half_width), int(center[1] + half_height))
    return *top_left, *bottom_right

def organizar_matriz(vetor, l=10, c=20):
    # Ordenar o vetor primeiro pelo segundo elemento (b) e depois pelo primeiro elemento (a)

    vetor_n = sorted(vetor, key=lambda x: x[1])  # Sort by 'y''
    vetor_n = np.array([sorted(vetor_n[i:i+c], key=lambda x: x[0]) for i in range(0, len(vetor_n), c)]).reshape(l,c,-1)

    # vetor_ordenado = np.array(sorted(vetor, key=lambda x: (x[1], x[0])))

    # # Dividir o vetor ordenado em 10 sublistas com 20 elementos cada, preservando a ordem interna
    # matriz = np.array([np.array_split(sublista, 3) for sublista in np.array_split(vetor_ordenado, 3)])

    return vetor_n

def distancia_euclidiana(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def pontos_faltando(array1, array2, limite):
    faltando = []
    for ponto1 in array1:
        presente = False
        for ponto2 in array2:
            if distancia_euclidiana(ponto1, ponto2) <= limite:
                presente = True
                break
        if not presente:
            faltando.append(ponto1)
    return faltando

if __name__ == "__main__":
    import numpy as np


    # Exemplo de uso
    array1 = np.array([(10,20), (30,40), (102, 23)])
    array2 = np.array([(11,22), (103,21)])

    limite = 5  # Ajuste o limite conforme necessário

    pontos_faltantes = pontos_faltando(array1, array2, limite)
    print("Pontos faltando em array2:", pontos_faltantes)
    array2 = np.concatenate((array2, pontos_faltantes))
    pontos_faltantes = pontos_faltando(array1, array2, limite)
    print("Pontos faltando em array2:", pontos_faltantes)

    # pontos_faltantes = pontos_faltando(array2, array1, limite)
    # print("Pontos faltando em array1:", pontos_faltantes)
