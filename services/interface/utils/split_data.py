import sys
from os.path import exists, join
from os import listdir, makedirs
from cv2 import selectROIs, destroyAllWindows, imread, imwrite, COLOR_BGR2GRAY, IMREAD_GRAYSCALE, cvtColor, resize, imwrite

from pickle import load, dump
from numpy import unique, append, array

from uuid import uuid1 as _id

import argparse

def applyRois(image, rois=[], expand=(0,0)):
    return ( image[y:y+h, x:x+w]  for x, y, w, h in rois if all(image[y:y+h, x:x+w].shape))

def createRois(image):
    rois = unique(selectROIs('Selecione as ROIs (Pressione Enter quando terminar)', image), axis=0)
    destroyAllWindows()
    return rois

def saveRois(path, rois):
    with open(path or './rois', 'wb') as file:
        return dump(rois, file)

def loadRois(path, expand=(0,0)):
    if exists(path):
        with open(path, 'rb') as file:
            rois = load(file)
            print(len(rois))
            rois = [((x+expand[0]), (y+expand[1]), w, h) for x, y, w, h in rois]
            return rois 
    return None
        
def loadFrames(path):
    images = []
    for filename in listdir(path):
        if filename.endswith('.jpg'):
            img = imread(join(path, filename))
            if img is not None:
                images.append(img)
    return images

def saveFrames(frames, path):
    if not exists(path):
        makedirs(path)
    for frame in frames:
        if frame is not None:
            imwrite(f'{path}/{_id().hex}.jpg', frame)

def normalizeData(frames, target_size=(128,128), gray=False):
    if gray:
        frames = [cvtColor(frame, COLOR_BGR2GRAY) for frame in frames]
    return [resize(frame, target_size) for frame in frames if all(frame.shape)]

def AdvancedRois(frame_path, w=55, h=77, r=0.7, _min=35, _max=40, group=50, show=True):
    import cv2
    import numpy as np
    from roi_viewer import calculate_corners, rounded_rectangle
    from uitls import calculate_average_coordinates

    frame = cv2.imread(frame_path)
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
    ctn = list(filter(lambda c: 800 < cv2.contourArea(c) < 8500, ctn))
    circles = [cv2.minEnclosingCircle(count) for count in ctn]
    
    coordinates_array = [circle[0] for circle in circles if _min < circle[1] < _max]

    coordinates_array = calculate_average_coordinates(coordinates_array, group)
    #print(len(coordinates_array))
    rois = [calculate_corners(coordinate, w, h) for coordinate in coordinates_array]
    for _idx, roi in enumerate(rois):
        x,y,w,h = roi
        rounded_rectangle(frame, (x,y), (x+w, y+h), r, (255, 255, 255))
        cv2.putText(frame, str(_idx), (int(x+(w/2)), int(y+h/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255,255,255))
        
    if show:
#        cv2.imshow('mk', MK )
        cv2.imshow('fr', frame )
#       if cv2.waitKey(0) == 27:
#           exit()
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
    parser.add_argument('-ADRR', '--AdvancedRois_R', type=int, help='Advanced ROIs')
    parser.add_argument('-ADRmin', '--AdvancedRois_min', type=int, help='Advanced ROIs')
    parser.add_argument('-ADRmax', '--AdvancedRois_max', type=int, help='Advanced ROIs')
    parser.add_argument('-SF', '--SaveFrames', type=str, help='Save frames to a directory')
    parser.add_argument('-ND', '--NormalizeData', action='store_true', help='Save frames to a directory')


    args = parser.parse_args(args)
    frames = None
    if args.LoadRois:
        rois = loadRois(args.LoadRois)
        print(f"Loaded {len(rois)}")

    if args.AdvancedRois:
        import cv2
        def n(x): return
        cv2.namedWindow('conf')
        cv2.resizeWindow('conf', (1200,600))
        cv2.createTrackbar('w','conf',args.AdvancedRois_W, 300,n)
        cv2.createTrackbar('h','conf',args.AdvancedRois_H, 300,n)
        cv2.createTrackbar('r','conf',args.AdvancedRois_R, 300,n)
        cv2.createTrackbar('min','conf',args.AdvancedRois_min, 300,n)
        cv2.createTrackbar('max','conf',args.AdvancedRois_max, 300,n)
        cv2.createTrackbar('group','conf', 10, 300,n)
        
        while cv2.waitKey(1) != 27:
            w = cv2.getTrackbarPos('w', 'conf')
            h = cv2.getTrackbarPos('h', 'conf')
            r = cv2.getTrackbarPos('r', 'conf')
            _min = cv2.getTrackbarPos('min', 'conf')
            _max = cv2.getTrackbarPos('max', 'conf')
            g = cv2.getTrackbarPos('group', 'conf')
            rois = AdvancedRois(args.AdvancedRois, w, h, r/10, _min, _max, g)

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
