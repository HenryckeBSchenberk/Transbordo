import math
import numpy as np
import cv2

def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    angulo_rad = np.arctan2(y1 - y2, x2 - x1)
    angulo_deg = np.degrees(angulo_rad)
    
    # Convertendo para um ângulo entre 0 e 360
    angulo_positivo = (angulo_deg + 360) % 360
    
    return angulo_positivo

class feature:
    def __init__(self, center, area, angle, anchor, mask=None) -> None:
        self.center = center
        self.area = area
        self.anchor = anchor
        self.angle = angle
        self.mask = mask
    
    def __str__(self) -> str:
        d = vars(self)
        d.pop('mask')
        return str(d)
    
    def __repr__(self) -> str:
        return self.__str__()


def get_features(i, max_area=5000, min_area=800)->[feature]:

    _, img = cv2.threshold(cv2.GaussianBlur(cv2.cvtColor(i, cv2.COLOR_BGR2GRAY), (15,15),0), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    c, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = np.array(list(filter(lambda x: min_area < cv2.contourArea(x) < max_area, c)))

    features = []
    for contour in c:
        M = cv2.moments(contour)

        Cx = int(M["m10"] / M["m00"])
        Cy = int(M["m01"] / M["m00"])

        ellipse = cv2.fitEllipse(contour)
        
        (x, y), (MA, ma), angle = ellipse
        angle_rad = np.deg2rad(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        A = (int(x - sin_angle * ma / 2), int(y + cos_angle * ma / 2))
        B = (int(x + sin_angle * ma / 2), int(y - cos_angle * ma / 2))                

        if get_distance(A, (Cx,Cy)) < get_distance(B, (Cx,Cy)):
            P2 = A
        else:
            P2 = B
        
        features.append(
            feature(
                (Cx,Cy),
                cv2.contourArea(contour),
                get_angle((Cx,Cy), P2),
                P2,
                img
            )
        )
    return features

def draw_features(i, features):
    for f in features:  
        i = cv2.bitwise_and(i,i,mask=f.mask)
        cv2.drawMarker(i, f.center, (255,255,255), 0, 5, 1, line_type=cv2.LINE_AA)
        cv2.putText(i,str(round(f.angle, 1)), (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255,255,255), 1)
        cv2.arrowedLine(i, f.center, f.anchor, (255,255,255), 2, line_type=cv2.LINE_AA)
    return i

def draw_ok_nok(i, flag, roi):
    if not flag:
        cv2.putText(i, 'NOK', (5,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
        cv2.rectangle(i, (0,0), roi[2::], (0,0,255), 3)
    else:
        cv2.putText(i, 'OK', (5,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
        cv2.rectangle(i, (0,0), roi[2::], (0,255,0), 3)
    return i