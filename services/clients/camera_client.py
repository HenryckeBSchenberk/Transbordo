import multiprocessing as mp
from multiprocessing.connection import Client
import cv2
def take_picture():
    with Client(("camera", 4000), authkey=b'secret') as c1, Client(("filter", 5000), authkey=b'secret') as c2:
        c2.send({'action':'roi', 'value':'200_roi'})
        # print('Filter echo:', c2.recv())
        c1.send({'action': 'take_picture'})
        # msg = c1.recv()
        # conn.close()
        # print('Image taken:', msg['status'])
    # cv2.imwrite('camera_client.jpg', msg['frame'])

if __name__ == "__main__":
    take_picture()
    # capture_process = mp.Process(target=take_picture)
    # capture_process.start()
    # capture_process.join()