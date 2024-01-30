import math
import numpy as np
import cv2
from sklearn.cluster import KMeans

KS = 15

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


def get_features(i, max_area=5000, min_area=100)->[feature]:

    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (KS,KS), 0)
    _, img = cv2.threshold(blur,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    c, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = filter(lambda x: min_area < cv2.contourArea(x) < max_area, c)

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

def divide_img_blocks(img, n_blocks=(5, 5)):
    horizontal = np.array_split(img, n_blocks[0])
    splitted_img = [np.array_split(block, n_blocks[1], axis=1) for block in horizontal]
    return np.asarray(splitted_img, dtype=np.ndarray).reshape(n_blocks).reshape(-1)

def calculate_roi_coordinates(roi_coords, n_blocks=(5, 5)):
    x_roi, y_roi, roi_width, roi_height = roi_coords
    block_width, block_height = roi_width // n_blocks[1], roi_height // n_blocks[0]

    row_indices, col_indices = np.indices(n_blocks)
    x = col_indices * block_width + x_roi
    y = row_indices * block_height + y_roi

    # Adjust the coordinates based on block index and size
    x = x.flatten()
    y = y.flatten()
    roi_coordinates = np.column_stack((x, y, block_width * np.ones_like(x), block_height * np.ones_like(y)))

    return roi_coordinates

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

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

if __name__ == '__main__':
    img = cv2.imread('./docs/asset/empty.jpg')
    calculate_roi_coordinates((68, 104, 1359, 913), (5,20))