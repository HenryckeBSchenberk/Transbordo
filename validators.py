from os import walk, path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from codetiming import Timer

from keras.models import (
        load_model
)

from numpy import (
    array,
    zeros_like,
    sum
)

default_threshold = lambda r: r[-1] > 0.5

steps = { 
    'presence': {
        'validator': './model/models/presence/best',
        'threshold': default_threshold,
        'qtd_models': 1
    },
    'orientation':{
        'validator':'./model/models/orientation_extra/best',
        'threshold':default_threshold,
        'qtd_models': 1
    },
}

class ModelWorker:
    def __init__(self, model):
        self.model = model

    def predict(self, frames):
        return self.model(array(frames)).numpy()
    
class Validator:
    def __init__(self, name, models_and_weights, max_workers=4):
        self.name = name
        if len(models_and_weights) == 1:
            self.model = model_and_weight[0][0]
            self.test = self.__test_with_single_model
        else:
            self.results = []
            self.lock = Lock()
            self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
            self.workers = [ModelWorker(model) for model, _ in models_and_weights]

            self.weights = [weight for _, weight in models_and_weights]
                # self.weights.append(weight)

            self.test = self.__test_with_multiple_models

    def _worker_predict(self, model, frames):
        return model.predict(frames)
    
    @Timer(name="Validation")
    def __test_with_single_model(self, frames):
        # return self.model(array(frames)).numpy() # Slow
        r = self.model.predict(array(frames), verbose=0)
        print(r)

        return r

    @Timer(name="Validation")
    def __test_with_multiple_models(self, frames):
        futures = []
        self.results = []  # Reset results for each test

        for worker in self.workers:
            future = self.thread_pool.submit(self._worker_predict, worker, frames)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            self.results.append(result)

        # Perform voting based on weights
        final_result = zeros_like(self.results[0])

        for i, result in enumerate(self.results):
            final_result += result * self.weights[i]

        final_result /= sum(self.weights)  # Normalize by sum of weights

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
    for root, dirs, files in walk(directory):
        for filename in files:
            if filename.endswith(".h5"):
                filepath = path.join(root, filename)
                epoch, loss, accuracy = extract_info_from_filename(filename)
                models_info.append((filepath, epoch, loss, accuracy))
                

    # Ordena os modelos com base no loss e precisão
    sorted_models = sorted(models_info, key=lambda x: (x[2], x[3]))

    # Pega os 'top_n' melhores modelos
    top_models = (load_model(model[0]) for model in sorted_models[:top_n])
    print(sorted_models[:top_n])
    return top_models

for step_name, step in steps.items():
    models = get_best_models(step['validator'], top_n=step['qtd_models'])
    model_and_weight = list(zip(models, [0.5, 0.3, 0.2]))
    step['validator'] = Validator(step_name, model_and_weight)