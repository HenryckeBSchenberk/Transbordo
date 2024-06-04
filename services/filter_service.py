import multiprocessing as mp
from interface.filter_interface import ImageFilter
from base_service import service                

from codetiming import Timer

@Timer(name="FilterImageProcess",  text="{name} demorou: {:.4f} segundos")
def callback(command, image_filter):
    action = command.get('action')
    match action:
        case 'image':
            image_filter.image = command.get('value')
            payload={'msg': "image received", 'to_output':False}
        case 'roi':
            image_filter.roi = command.get('value')
            payload={'msg': "roi received", 'to_output':False}
        case _:
            pass
    if image_filter.has_unread_result:
        payload = {'action':'predict', 'value': image_filter.get_info(), 'to_output':True}
    return payload

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Client script for capturing images and sending commands to the model server.")
    parser.add_argument('--service_host', type=str, required=False, help="(HOST) where the this service are.", default="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="5000")
    parser.add_argument('--output_host', type=str, required=False, help="(HOST) where the information goes.", default="tensorflow")
    parser.add_argument('--output_port', type=str, required=False, help="(PORT) where the information goes.", default="6000")
    
    args = parser.parse_args()
    callback_args=(ImageFilter(),)
    server_process = mp.Process(target=service, args=((args.service_host, int(args.service_port)), (args.output_host, int(args.output_port)), callback, *callback_args))

    print(f"{__file__} started.")
    server_process.start()
    server_process.join()
    print(f"{__file__} stopped.")