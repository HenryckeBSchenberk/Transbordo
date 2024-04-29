import argparse
import importlib
from codetiming import Timer
from cv2 import (
    imread,
    COLOR_BGR2RGB,
    cvtColor,
)

from split_data import (
    loadRois,
    normalizeData,
    applyRois
)

from uitls import (
    get_features,
    draw_features,
    draw_ok_nok,
    draw_roi
)
    
class frame:
    def __init__(self, frame_orignal, frame_validation, roi, **kwargs) -> None:
        self.frame_original = frame_orignal
        self.frame_validation = frame_validation
        self.roi = roi

        for k,v in kwargs.items():
            setattr(self, k, v)
        
    def update(self, attr, value):
        setattr(self, attr, value)
        return value
    
    def __str__(self) -> str:
        d = vars(self)
        d.pop('frame_original')
        d.pop('frame_validation')
        d.pop('roi')

        return str(d)

    def __repr__(self) -> str:
        return self.__str__()

def perform_validation(frames, frame_info, thr_function, validator):
    result = validator.test(frames)
    new_frames, new_frame_info = [
        ft.frame_validation for (ft, r) in zip(frame_info, result) if ft.update(validator.name, thr_function(r))
    ], filter(lambda info: getattr(info, validator.name), frame_info) 

    return new_frames, new_frame_info

def perform_steps(frames, frame_info, steps, keys=False):
    if keys != [] and frames != []:
        keys = keys or list(steps.keys())
        step = steps[keys.pop(0)]
        thf = step['threshold']
        validator = step['validator']
        nf, ni = perform_validation(frames, frame_info, thf, validator)
        return perform_steps(nf, ni, steps, keys)

def insertFrames(rois, frames, image):
    for (x, y, w, h), frame in zip(rois, frames):
        image[y:y+h, x:x+w] =  frame

@Timer(name="FullProcess",  text="{name} demorou: {:.4f} segundos")
def validate(_frame, _steps, _roi, expand=(0,0), _show=False):
    if _frame is not None:

        img = _frame
        original = img.copy()
        original2 = img.copy()
        rois = _roi

        frames2draw = list(applyRois(img, rois, expand))
        frames2validade = normalizeData(frames2draw)
        frame_info = [frame(fo, fd, r, presence=False, orientation=False) for fo, fd, r in zip(frames2draw, frames2validade, rois)]
        perform_steps(frames2validade, frame_info, _steps)

        frame_info = sorted(frame_info, key=lambda x: x.roi[1])  # Sort by 'y'
        # Sort within each group of 20 by 'x'
        frame_info = [sorted(frame_info[i:i+20], key=lambda x: x.roi[0]) for i in range(0, len(frame_info), 20)]
        frame_info = [item for sublist in frame_info for item in sublist] 

        for _frame in frame_info:
            roi = _frame.roi
            x,y,w,h = roi
            i = _frame.frame_original
            draw_roi(i, roi)
            original[y:y+h, x:x+w] = i

            if _frame.presence:
                
                features = get_features(i)
                _frame.update('features', features)

                i = draw_features(i, features)
                draw_ok_nok(i, _frame.orientation, roi)
                original[y:y+h, x:x+w] = i
                continue

        return frame_info, original, original2

    raise AttributeError('An image/frame should be provided to this function analise')

_default_rois = lambda: loadRois('./rois')
_default_steps = lambda: importlib.import_module('validators').steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and ROIs.')
    parser.add_argument('-P', '--path', type=str, help='Path for image raw image')
    parser.add_argument('-R', '--roi', type=str, help="Path for 'roi' file", default='./rois')
    parser.add_argument('-S', '--show', type=int, help='Show final result with matplotlib')
    parser.add_argument('-M', '--models', type=str, help='python package file that describe a steps:{<step_name>:{model:<model_path>,threshold:<lambda_validator>}, ...}', default='validators')
    args = parser.parse_args()
    
    _frame = imread(args.path)
    _rois = loadRois(args.roi)

    steps = importlib.import_module(args.models).steps
    _info, _new, _c = validate(_frame=_frame, _roi=_rois, _show=args.show, _steps=steps.copy())

    presence =    [ frame.presence for frame in _info ]
    orientation = [ frame.orientation for frame in _info ]
    # enviar_valores_para_clp(100, presence)
    # print(presence,'\n', orientation)
