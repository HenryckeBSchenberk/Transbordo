
import cv2 
import numpy as np
import auto_roi.utils as utils

def draw(img, detected_circles, _mask=None, _r=0.7, _w=55,_h=77, group_distance=40, p=3, color=(0,0,0)):
    # Draw circles that are detected.
    mask = _mask if _mask is not None else np.zeros(img.shape[:2], dtype="uint8")
    if detected_circles is not None: 
        _centers, _x, _y = centers(detected_circles, p, group_distance)
        detected_circles = np.uint16(np.around(detected_circles))
        _centers = np.uint16(np.around(_centers))

        rois = [utils.calculate_corners(coordinate, _w, _h) for coordinate in _centers[0, :]]
        for pt, roi in zip(_centers[0, :], rois) :
            a, b, r = pt[0], pt[1], int(_w/2) 

            x,y,w,h = roi
            utils.rounded_rectangle(img, (x,y), (x+w, y+h), _r, color)
            utils.rounded_rectangle(mask, (x,y), (x+w, y+h), _r)
            ctn, _ = cv2.findContours(mask,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.fillPoly(mask, pts =ctn, color=(255,255,255))

            cv2.circle(img, (a, b), r, (0, 255, 0), 1) 
            cv2.circle(img, (a, b), 1, (0, 0, 255), 1)
            cv2.line(img, (a,b), (a, int(b+_y)), (255,0,0))
            cv2.line(img, (a,b), (int(a+_x), b), (255,255,0))

    img = cv2.bitwise_and(img, img, mask=cv2.GaussianBlur(mask, (15,15), 1))
    return _centers, rois, img, mask
 

def detect(img, minDist=23, param1=48, param2=17, minRadius=24, maxRadius=27):
    # Convert to grayscale. 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3)) 
    
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                    cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=param1, 
                param2=param2 , minRadius=minRadius, maxRadius=maxRadius) 
    return detected_circles
  
def detect_2(img, _min, _max, group_distance, a_min=700, a_max=7500):
    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    B = cv2.GaussianBlur(G, (5,5), 0)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,7))

    MK = cv2.adaptiveThreshold(B,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,11,2)
    
    ctn, _ = cv2.findContours(MK, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctn = list(filter(lambda c: a_min < cv2.contourArea(c) < a_max, ctn))
    circles = [cv2.minEnclosingCircle(count) for count in ctn]
    rectangles = [cv2.minAreaRect(count) for count in ctn]
    rectangles = [[(((x+w)/2), ((y+h)/2)), r] for (x, y), (w, h), r in rectangles]
    coordinates_array_C = [circle[0] for circle in circles if _min < circle[1] < _max]
    coordinates_array_C = utils.calculate_average_coordinates(coordinates_array_C, group_distance or 1)
    return coordinates_array_C, MK

def centers(detected_circles, p =3, group_d=40):
    d = detected_circles[0]

    d = d.reshape(-1, len(d), p)

    y = np.array([utils.calculate_average_coordinates(d[0, :, :p], group_d)])
    y = np.array([utils.calculate_average_coordinates(y[0, :, :p], group_d)])

    y2 = y[0].reshape(-1,p)


    _x = np.sort((y2[:,0]).flatten())
    _y = np.sort((y2[:,1]).flatten())
    x_diff = np.diff(_x)
    y_diff = np.diff(_y)


    x_diff = x_diff[x_diff>10]
    y_diff = y_diff[y_diff>10]

    x_values = x_diff.max()
    y_values = y_diff.max()

    return y, x_values, y_values
 

def PA(a1, n, r):
    return a1 +(n-1)*r
