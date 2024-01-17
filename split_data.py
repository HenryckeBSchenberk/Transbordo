import sys
from os.path import exists, join
from os import listdir
from cv2 import selectROIs, destroyAllWindows, imread, imwrite, COLOR_BGR2GRAY, IMREAD_GRAYSCALE, cvtColor, resize, imwrite

from pickle import load, dump
from numpy import unique, append, array

from uuid import uuid1 as _id

import argparse

def applyRois(image, rois):
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
        if filename.endswith('.jpg'):
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
    

def main(args):
    parser = argparse.ArgumentParser(description='Process images and ROIs.')
    parser.add_argument('-CR', '--CreateRois', type=str, help='Create ROIs for an image')
    parser.add_argument('-AR', '--ApplyRois', type=str, help='Apply ROIs to an image', default="./")
    parser.add_argument('-LF', '--LoadFrames', type=str, help='Load frames from a directory')
    parser.add_argument('-LR', '--LoadRois', type=str, help='Load ROIs from a file')
    parser.add_argument('-SR', '--SaveRois', type=str, help='Save ROIs to a file')
    parser.add_argument('-SF', '--SaveFrames', type=str, help='Save frames to a directory')
    parser.add_argument('-ND', '--NormalizeData', action='store_true', help='Save frames to a directory')

    args = parser.parse_args(args)

    if args.LoadRois:
        rois = loadRois(args.LoadRois)

    if args.CreateRois:
        rois = createRois(imread(args.CreateRois))
        if args.SaveRois:
            saveRois(args.SaveRois, rois)

    if args.ApplyRois or args.LoadFrames:
        if args.LoadFrames:
            frames = loadFrames(args.LoadFrames)
            print(f"loaded {len(frames)} frames.")
            if args.ApplyRois:
                i= 0
                _frames = []
                for f in frames:
                    for ff in applyRois(f, rois):
                        _frames.append(ff)
                frames = _frames
        else:
            frames = applyRois(imread(args.ApplyRois), rois)
        if args.NormalizeData:
            frames = normalizeData(frames)
        if args.SaveFrames:
            saveFrames(frames, args.SaveFrames)
        else:
            from cv2 import imshow, waitKey
            for frame in frames:
                imshow('roi', frame)
                waitKey(100)

            destroyAllWindows()

    
if __name__ == '__main__':
    main(sys.argv[1:])
