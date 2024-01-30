import sys
from os.path import exists, join
from os import listdir
from cv2 import selectROIs, destroyAllWindows, imread, imwrite, COLOR_BGR2GRAY, IMREAD_GRAYSCALE, cvtColor, resize, imwrite

from pickle import load, dump
from numpy import unique, append, array

from uuid import uuid1 as _id

import argparse

def applyRois(image, rois=[], expand=(0,0)):
        # rois = [((x-expand[0]), (y-expand[1]), (w+(expand[0]*2)), (h+(expand[1]*2))) for x, y, w, h in rois]
        return (
             image[y:y+h, x:x+w]  for x, y, w, h in rois
        )

def createRois(image):
    rois = unique(selectROIs('Selecione as ROIs (Pressione Enter quando terminar)', image), axis=0)
    destroyAllWindows()
    return rois

def saveRois(path, rois):
    with open(path or './rois', 'wb') as file:
        return dump(rois, file)

def loadRois(path):
    if exists(path):
        with open(path, 'rb') as file:
            return  load(file)
        
def loadFrames(path):
    images = []
    for filename in listdir(path):
        if filename.endswith('.png'):
            img = imread(join(path, filename))
            if img is not None:
                images.append(img)
    return images

def saveFrames(frames, path):
    if not exists(path):
        raise AssertionError(f"{path} not exist!")
    for frame in frames:
        if frame is not None:
            imwrite(f'{path}/{_id().hex}.jpg', frame)

def normalizeData(frames, target_size=(128,128), gray=False):
    if gray:
        frames = [cvtColor(frame, COLOR_BGR2GRAY) for frame in frames]
    return [resize(frame, target_size) for frame in frames]

def AdvancedRois(frame_path, w=55, h=77, r=0.7, _min=35, _max=40, show=True):
    import cv2
    import numpy as np
    from roi_viewer import calculate_corners, rounded_rectangle
    from uitls import calculate_average_coordinates

    frame = cv2.imread(frame_path) if isinstance(frame_path, str) else frame_path
    drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    B = cv2.GaussianBlur(G, (5,5), 0)
    
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,7))

    MK = cv2.adaptiveThreshold(B,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,11,2)
    # MK = cv2.morphologyEx(MK, cv2.MORPH_OPEN, k, iterations=1)
    # MK = cv2.morphologyEx(MK, cv2.MORPH_ERODE, k, iterations=1)
    # MK = cv2.morphologyEx(MK, cv2.MORPH_CLOSE, k, iterations=1)
    ctn, _ = cv2.findContours(MK, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctn = list(filter(lambda c: 700 < cv2.contourArea(c) < 7500, ctn))
    circles = [cv2.minEnclosingCircle(count) for count in ctn]
    
    coordinates_array = [circle[0] for circle in circles if _min < circle[1] < _max]

    coordinates_array = calculate_average_coordinates(coordinates_array, 15)
    print(len(coordinates_array))
    rois = [calculate_corners(coordinate, w, h) for coordinate in coordinates_array]
    for _idx, roi in enumerate(rois):
        x,y,w,h = roi
        rounded_rectangle(drawing, (x,y), (x+w, y+h), r, (255, 255, 255))
        cv2.putText(drawing, str(_idx), (int(x+(w/2)), int(y+h/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255))
        
    if show:
        cv2.imshow('mk', MK )
        cv2.imshow('fr', drawing )
        if cv2.waitKey(0) == 27:
            exit()
    return rois
    


def main(args):
    parser = argparse.ArgumentParser(description='Process images and ROIs.')
    parser.add_argument('-CR', '--CreateRois', type=str, help='Create ROIs for an image')
    parser.add_argument('-AR', '--ApplyRois', type=str, help='Apply ROIs to an image', default="./")
    parser.add_argument('-LF', '--LoadFrames', type=str, help='Load frames from a directory')
    parser.add_argument('-LR', '--LoadRois', type=str, help='Load ROIs from a file')
    parser.add_argument('-SR', '--SaveRois', type=str, help='Save ROIs to a file')
    parser.add_argument('-ADR', '--AdvancedRois', type=str, help='Advanced ROIs')
    parser.add_argument('-ADRW', '--AdvancedRois_W', type=int, help='Advanced ROIs')
    parser.add_argument('-ADRH', '--AdvancedRois_H', type=int, help='Advanced ROIs')
    parser.add_argument('-ADRR', '--AdvancedRois_R', type=float, help='Advanced ROIs')
    parser.add_argument('-ADRmin', '--AdvancedRois_min', type=float, help='Advanced ROIs')
    parser.add_argument('-ADRmax', '--AdvancedRois_max', type=float, help='Advanced ROIs')
    parser.add_argument('-SF', '--SaveFrames', type=str, help='Save frames to a directory')
    parser.add_argument('-ND', '--NormalizeData', action='store_true', help='Save frames to a directory')


    args = parser.parse_args(args)
    frames = None
    if args.LoadRois:
        rois = loadRois(args.LoadRois)
        print(f"Loaded {len(rois)}")

    if args.AdvancedRois:
        rois = AdvancedRois(args.AdvancedRois, args.AdvancedRois_W, args.AdvancedRois_H, args.AdvancedRois_R, args.AdvancedRois_min, args.AdvancedRois_max)

    if args.CreateRois:
        rois = createRois(imread(args.CreateRois))

    if args.SaveRois:
        saveRois(args.SaveRois, rois)

    if args.ApplyRois or args.LoadFrames:
            
        if args.LoadFrames:
            print(args.LoadFrames)
            frames = loadFrames(args.LoadFrames)
            print(f"loaded {len(frames)} frames.")
            if args.ApplyRois:
                i= 0
                _frames = []
                for f in frames:
                    for ff in applyRois(f, rois):
                        _frames.append(ff)
                frames = _frames
        elif args.ApplyRois != "./":
            frames = applyRois(imread(args.ApplyRois), rois)
        if args.NormalizeData:
            frames = normalizeData(frames)
        if args.SaveFrames:
            saveFrames(frames, args.SaveFrames)
        elif frames:
            from cv2 import imshow, waitKey
            for frame in frames:
                imshow('roi', frame)
                waitKey(100)

            destroyAllWindows()

    
if __name__ == '__main__':
    main(sys.argv[1:])
