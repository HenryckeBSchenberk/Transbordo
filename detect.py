import threading
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import argparse
from keras.models import (
        load_model
)

import importlib
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
    draw_ok_nok
)
    
class ModelWorker:
    def __init__(self, model):
        self.model = model

    def predict(self, frames):
        return self.model.predict(np.array(frames))

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

class Validator:
    def __init__(self, name, models_and_weights, max_workers=4):
        self.name = name
        self.models = []
        self.weights = []
        self.results = []
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.workers = [ModelWorker(model) for model, _ in models_and_weights]

        for model, weight in models_and_weights:
            self.models.append(model)
            self.weights.append(weight)

    def _worker_predict(self, model, frames):
        return model.predict(frames)

    def test(self, frames):
        futures = []
        self.results = []  # Reset results for each test

        for worker in self.workers:
            future = self.thread_pool.submit(self._worker_predict, worker, frames)
            futures.append(future)

        for future in futures:
            result = future.result()
            self.results.append(result)

        # Perform voting based on weights
        final_result = np.zeros_like(self.results[0])

        for i, result in enumerate(self.results):
            final_result += result * self.weights[i]

        final_result /= np.sum(self.weights)  # Normalize by sum of weights

        return final_result.round().astype(float)  # Round to get bina

def extract_info_from_filename(filename):
    parts = filename.split('-')
    epoch = int(parts[1])
    loss = float(parts[2])
    accuracy = float(parts[3].split('.h5')[0])  # Remove '.h5' extension and convert to float
    return epoch, loss, accuracy

def get_best_models(directory, top_n=3):
    models_info = []

    # Percorre recursivamente os arquivos na pasta 'directory'
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".h5"):
                filepath = os.path.join(root, filename)
                epoch, loss, accuracy = extract_info_from_filename(filename)
                models_info.append((filepath, epoch, loss, accuracy))
                

    # Ordena os modelos com base no loss e precisão
    sorted_models = sorted(models_info, key=lambda x: (x[2], x[3]))

    # Pega os 'top_n' melhores modelos
    top_models = (load_model(model[0]) for model in sorted_models[:top_n])
    print(sorted_models[:top_n])
    return top_models

def perform_validation(frames, frame_info, thr_function, validator):
    result = validator.test(frames)
    new_frames, new_frame_info = [
        ft.frame_validation for (ft, r) in zip(frame_info, result) if ft.update(validator.name, thr_function(r))
    ], filter(lambda info: getattr(info, validator.name), frame_info) 

    return new_frames, new_frame_info

def perform_steps(frames, frame_info, steps):
    keys = list(steps.keys())
    if len(keys)>0:
        step = steps.pop(keys.pop(0))
        thf = step['threshold']
        validator = step['validator']
        nf, ni = perform_validation(frames, frame_info, thf, validator)
        return perform_steps(nf, ni, steps)

def insertFrames(rois, frames, image):
    for (x, y, w, h), frame in zip(rois, frames):
        image[y:y+h, x:x+w] =  frame


def validate(_frame, _steps, _roi, _show):
    if _frame is not None:

        img = _frame
        original = img.copy()
        rois = _roi

        frames2draw = list(applyRois(img, rois))
        frames2validade = normalizeData(frames2draw)
        frame_info = [frame(fo, fd, r, presence=False, orientation=False) for fo, fd, r in zip(frames2draw, frames2validade, rois)]
        perform_steps(frames2validade, frame_info, _steps)

        if _show:
            import matplotlib.pyplot as plt
            cols = _show
            rows = (len(frames2draw) + cols - 1) // cols

        frame_info = sorted(frame_info, key=lambda x: x.roi[1])  # Sort by 'y'
        # Sort within each group of 20 by 'x'
        frame_info = [sorted(frame_info[i:i+20], key=lambda x: x.roi[0]) for i in range(0, len(frame_info), 20)]
        frame_info = [item for sublist in frame_info for item in sublist] 

        for idx,  _frame in enumerate(frame_info):
            roi = _frame.roi
            x,y,w,h = roi
            i = _frame.frame_original
            original[y:y+h, x:x+w] = i
            if _show:
                ax = plt.subplot(rows, cols, idx+1)
                ax.set_xticks([])
                ax.set_yticks([])

            if _frame.presence:
                
                features = get_features(i)
                _frame.update('features', features)

                if _show:
                    i = draw_features(i, features)
                    draw_ok_nok(i, _frame.orientation, roi)
                    original[y:y+h, x:x+w] = i
                    plt.imshow(cvtColor(i, COLOR_BGR2RGB))
                    continue
            if _show:
                plt.imshow(np.zeros(i.shape, dtype=np.uint8), cmap='gray')

        if _show:
            plt.show()

        return frame_info, original

    raise AttributeError('An image/frame should be provided to this function analise')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images and ROIs.')
    parser.add_argument('-P', '--path', type=str, help='Path for image raw image')
    parser.add_argument('-R', '--roi', type=str, help="Path for 'roi' file", default='./rois')
    parser.add_argument('-S', '--show', type=int, help='Show final result with matplotlib')
    parser.add_argument('-M', '--models', type=str, help='python package file that describe a steps:{<step_name>:{model:<model_path>,threshold:<lambda_validator>}, ...}', default='validators')
    args = parser.parse_args()
    
    _frame = imread(args.path)
    _rois = loadRois(args.roi)

    vd = importlib.import_module(args.models)
    steps = vd.steps

    for step_name, step in steps.items():
        models = get_best_models(step['validator'], top_n=1)
        model_and_weight = list(zip(models, [0.5, 0.3, 0.2]))
        step['validator'] = Validator(step_name, model_and_weight)

    validate(_frame=_frame, _roi=_rois, _show=args.show, _steps=steps)