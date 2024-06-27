import cv2
import multiprocessing as mp
import numpy as np
from multiprocessing.connection import Client
from codetiming import Timer

from interface.utils.split_data import (
    loadRois,
    normalizeData,
    applyRois
)
from interface.filter_interface import ImageFilter

t = Timer(name="Requisição")
def capture_and_send_commands(model_path):
    conn = Client(("tensorflow", 6000), authkey=b'secret')
    conn.send({'action': 'train'})
    # f = ImageFilter()
    # _frame = cv2.imread("/app/img.jpg")
    # f.roi= "200_roi"
    # f.image = _frame
    # if model_path is not None:
        # conn.send({'action': 'load_model', 'model_path': f'/app/{model_path}'})
    #     print(conn.recv())

    # t.start()
    # conn.send({'action':'predict', 'value': f.get_info(), 'to_output':True})
    # t.stop()
    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Client script for capturing images and sending commands to the model server.")
    parser.add_argument('--model_path', type=str, required=False, help="Path to the TensorFlow model to be loaded by the server.")

    args = parser.parse_args()

    capture_process = mp.Process(target=capture_and_send_commands, args=(args.model_path,))
    capture_process.start()
    capture_process.join()

