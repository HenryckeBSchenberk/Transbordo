import multiprocessing as mp
from threading import Thread, Event

from interface.camera_interface import (
    Basler,
    OpenCV,
    cv2,
)
import numpy as np
from base_service import service
from codetiming import Timer

@Timer(name="CaptureImageProcess",  text="{name} demorou: {:.4f} segundos")
def callback(command, cam):
    action = command.get('action')
    match action:
        case 'take_picture':
            status, frame = cam.read()
            cv2.imwrite('/app/raw.jpg', frame)
            return {'action':'image', 'status':status, 'value':frame, 'to_output':True}
        case _:
            return {'status':False, 'frame':None, 'to_output':False}

stop_signal = Event()
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client script for capturing images and sending commands to the model server.")
    parser.add_argument('--service_host', type=str, required=False, help="(HOST) where the this service are.", default="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="4000")
    parser.add_argument('--output_host', type=str, required=False, help="(HOST) where the information goes.", default="filter")
    parser.add_argument('--output_port', type=str, required=False, help="(PORT) where the information goes.", default="5000")

    args = parser.parse_args()
    cam = Basler(configuration="cameraSettings", stop_signal=stop_signal)
    print(cam.read_raw())
    callback_args = (cam, )
    kwargs={'stop_signal':stop_signal, }
    server_process = Thread(target=service, args=((args.service_host, int(args.service_port)), (args.output_host, int(args.output_port)), callback, *callback_args), kwargs=kwargs, daemon=True)

    print(f"{__file__} started.")
    server_process.start()
    cam.thread.join()
    if stop_signal.is_set():
        exit(1)
    server_process.join()
    print(f"{__file__} stopped.")
