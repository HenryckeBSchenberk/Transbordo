import cv2
import numpy as np

def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA, from_zero=False):

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

if __name__ == '__main__':
    img=cv2.imread("D:/Images/M2/nok/Bigger/Image__2024-01-26__17-22-09.png")

    source_window = 'Source'
    cv2.namedWindow(source_window)

    class config:
        def __init__(self, top_left, size, radius) -> None:
            self.x, self.y = top_left
            self.x2, self.y2 = size
            self.radius = radius
        
        def export(self):
            return {
                "top_left":(self.x, self.y),
                "bottom_right":(self.x2, self.y2),
                "radius":self.radius/10
            }

    c = config((10,10), (100,100), 0.3)

    cv2.createTrackbar('Origin X', source_window, 69, img.shape[0]-1, lambda val: c.__setattr__('x', val))
    cv2.createTrackbar('Origin Y', source_window, 121, img.shape[1]-1, lambda val: c.__setattr__('y', val))
    cv2.createTrackbar('Size X', source_window, 124, img.shape[0], lambda val: c.__setattr__('x2', val))
    cv2.createTrackbar('Size Y', source_window, 196, img.shape[1], lambda val: c.__setattr__('y2', val))
    cv2.createTrackbar('radius', source_window, 7, 10, lambda val: c.__setattr__('radius', val))

    while cv2.waitKey(1) != 27:
        dd = c.export()
        draw = img.copy()
        cv2.rectangle(draw, dd['top_left'], dd['bottom_right'], (0,0,255))
        draw = rounded_rectangle(draw, color=(255,0,255), **dd)
        draw = rounded_rectangle(draw, color=(255,255,255), from_zero=True, **dd)
        
        w, h = 124-69, 196-121
        center = (w/2, h/2)
        (x1,y1, x2,y2) = calculate_corners(center, w, h, False)
        tl = (x1,y1)
        br = (x2,y2)
        draw = rounded_rectangle(draw, color=(0,0,255), top_left=tl, bottom_right=br, radius=0.7)

        cv2.imshow('draw', draw) #73,121,130,260,0.2