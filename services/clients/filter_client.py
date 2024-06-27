import multiprocessing as mp
from multiprocessing.connection import Client
import cv2

def get_and_filter():
    print('a')
    conn_camera = Client(("camera", 4000), authkey=b'secret')
    conn_filter = Client(("filter", 5000), authkey=b'secret')
    print('b')
    
    conn_filter.send({'action':'roi', 'value':'200_roi'})
    conn_camera.send({'action': 'take_picture'})
    # img = conn_camera.recv()

    
    # if img['status']:
    #     cv2.imwrite('filter_camera_client.jpg', img['value'])
        # conn_filter.send({'action':'image', 'value':img['value']})
    
    # msg = conn_filter.recv()
    # print(img['status'], msg is not None)
    conn_filter.close()
    conn_camera.close()

if __name__ == "__main__":
    # take_picture()
    capture_process = mp.Process(target=get_and_filter)
    capture_process.start()
    capture_process.join()