import numpy as np

def calcular_angulo(ponto1, ponto2):
    x1, y1 = ponto1
    x2, y2 = ponto2
    
    angulo_rad = np.arctan2(y1 - y2, x2 - x1)
    angulo_deg = np.degrees(angulo_rad)
    
    # Convertendo para um ângulo entre 0 e 360
    angulo_positivo = (angulo_deg + 360) % 360
    
    return angulo_positivo

# Exemplo de uso
ponto1 = (5, 0)
ponto2 = (0, 5)

angulo = calcular_angulo(ponto1, ponto2)
print("O ângulo entre os pontos é:", angulo, "graus")