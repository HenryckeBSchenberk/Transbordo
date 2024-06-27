from base_service import service
import multiprocessing as mp
import numpy as np
from interface.interpreter_interface import create_relation, relation
import matplotlib.pyplot as plt
import cv2
from codetiming import Timer

def update(obj):
    return obj.update_values(**vars(relation(np.argmax(obj.prediction, axis=-1), classes=['presence', 'orientation'], answer=[False, False])))

def draw_ok_nok(i, flag, roi):
    if not flag:
        cv2.putText(i, 'NOK', (5,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 1)
        cv2.rectangle(i, (0,0), roi[2::], (0,0,255), 3)
    else:
        cv2.putText(i, 'OK', (5,15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
        cv2.rectangle(i, (0,0), roi[2::], (0,255,0), 3)
    return i

def draw_roi(i, roi, idx):
    cv2.rectangle(i, (0,0), roi[2::], (255,255,255), 3)
    cv2.putText(i, str(idx), (5,35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
    return i

def draw(result):
    canvas = np.zeros((1080, 1440, 3), dtype=np.uint8)
    for idx, _inf in enumerate(result):
        roi = _inf.roi
        x, y, w, h = roi
        i = _inf.original_data
        i = draw_roi(i, roi, idx)
        canvas[y:y+h, x:x+w] = i
        if _inf.presence:
            i = draw_ok_nok(i, _inf.orientation, roi)
            canvas[y:y+h, x:x+w] = i
    cv2.imwrite('process.jpg', canvas)


@Timer(name="InterpretPredctionsProcess",  text="{name} demorou: {:.4f} segundos")
def callback(command, *args):
    action = command.get('action')

    match action:
        case 'interpret':
            input_value = command.get('value')
            # print(input_value)
            result = list(map(update, input_value))
            frame_info = sorted(result, key=lambda x: x.roi[1])  # Sort by 'y'
            # Sort within each group of 20 by 'x'
            frame_info = [sorted(frame_info[i:i+20], key=lambda x: x.roi[0]) for i in range(0, len(frame_info), 20)]
            frame_info = [item for sublist in frame_info for item in sublist]
            draw(frame_info)

            payload = {}


            for k in ['presence', 'orientation']:
                payload[k] = [ getattr(info, k, False) for info in frame_info ][::-1] 
            # print(payload)

            s = payload | {'action': 'communicate', 'to_output':True}
            return s
                


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client script for capturing images and sending commands to the model server.")
    parser.add_argument('--service_host', type=str, required=False, help="(HOST) where the this service are.", default="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="5000")
    parser.add_argument('--output_host', type=str, required=False, help="(HOST) where the information goes.", default="tensorflow")
    parser.add_argument('--output_port', type=str, required=False, help="(PORT) where the information goes.", default="6000")
    
    args = parser.parse_args()

    callback_args = (None,)
    server_process = mp.Process(target=service, args=((args.service_host, int(args.service_port)), (args.output_host, int(args.output_port)), callback, *callback_args))

    print(f"{__file__} started.")
    server_process.start()
    server_process.join()
    print(f"{__file__} stopped.")