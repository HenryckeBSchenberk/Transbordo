import multiprocessing as mp
from interface.clp_interface import CLP
from multiprocessing.connection import Client
from base_service import service
from time import sleep
from socket import gaierror

CAMERA_TRIGGER_REG=66
CAMERA_TRIGGER_OK=67
MODEL_REGISTER=68
OK_RECALIBRATE_REG=69

from codetiming import Timer

from threading import Event, Thread
stop_signal = Event()

@Timer(name="TCPCommunicationProcess",  text="{name} demorou: {:.4f} segundos")
def callback(command, clp):
    action = command.get('action')
    match action:
        case 'communicate':     # Escreve os valores identificados nas mememorisa correspondetnes do clp
            clp.write_group_data(804, command.get('presence', []), 20)
            clp.write_group_data(300, command.get('orientation', []), 20)
            clp.client.write_multiple_coils(CAMERA_TRIGGER_OK, [True])

    return {}


@Timer(name="CameraTrigger",  text="{name} demorou: {:.4f} segundos")
def callback_camera_trigger(clp, coil_state):
    print("gotcha")
    try:
        with Client(("camera", 4000), authkey=b'secret') as output_conn:
            output_conn.send({'action': 'take_picture'})
    except (gaierror, ConnectionRefusedError) as e:
        print(e)

@Timer(name="RoiSelector",  text="{name} demorou: {:.4f} segundos")
def callback_filter_roi(clp, coil_state):
    try:
        with Client(("filter", 5000), authkey=b'secret') as output_conn:
            output_conn.send({'action':'roi', 'value':f'{200 if coil_state else 100}_roi'})
    except (gaierror, ConnectionRefusedError) as e:
        print(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client script for capturing images and sending commands to the model server.")
    parser.add_argument('--service_host', type=str, required=False, help="(HOST) where the this service are.", default="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="8000")
    parser.add_argument('--output_host', type=str, required=False, help="(HOST) where the information goes.", default="")
    parser.add_argument('--output_port', type=str, required=False, help="(PORT) where the information goes.", default="")
    
    args = parser.parse_args()
    clp = CLP('192.168.2.5', 502, 1, True, stop_signal=stop_signal)
    callback_args=(clp, )
    kwargs={'stop_signal':stop_signal, }
    print(clp.client)
    clp.register(CAMERA_TRIGGER_REG, 'RISING', callback_camera_trigger)
    clp.register(MODEL_REGISTER, 'CHANGE', callback_filter_roi)

    print(args.service_host, int(args.service_port))
    server_process = Thread(target=service, args=((args.service_host, int(args.service_port)), (args.output_host, int(args.output_port)), callback, *callback_args), kwargs=kwargs, daemon=True)

    print(f"{__file__} started.")
    server_process.start()
    clp.thread.join()
    if stop_signal.is_set():
        exit(1)
    server_process.join()
    print(f"{__file__} stopped.")