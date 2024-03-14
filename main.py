from modbus import PLC, Camera, CAMERA_TRIGGER_REG, CAMERA_TRIGGER_OK,enviar_valores_para_clp
from detect import validate, _default_rois, _default_steps, loadRois
from codetiming import Timer
import cv2
import pickle
SHOW = True
mx, my = pickle.load(open('mp_xy.pkl', 'rb')) 

# @Timer(name="Main validation process")
def process(camera):
    should_trigger = PLC.read_coils(CAMERA_TRIGGER_REG, 1)[0]
    if should_trigger:
        print("m60-m70 status: ", PLC.read_coils(60,10))
        #PLC.write_single_coil(CAMERA_TRIGGER_OK, False)
        #PLC.write_single_coil(CAMERA_TRIGGER_REG, False)
        for _ in range(3):
            _info, _frame = camera.read()
            _frame = cv2.remap(_frame, mx, my, cv2.INTER_LINEAR)
            
        if _info:
            if PLC.read_coils(68, 1)[0]:
                _default_rois=_default_200rois
            else:
                _default_rois=_default_100rois
            return validate(_frame,  _default_steps.copy(), _default_rois, (10,10), 20)
    PLC.write_multiple_coils(CAMERA_TRIGGER_REG,[False,False])
    return None, None, None
if __name__ == '__main__':
    _default_steps = _default_steps()
    _default_200rois = loadRois("200_roi",(10,0))
    _default_100rois = loadRois("100_roi",(0,0))
    

    camera = Camera(False)
    #camera = Camera(False, "/home/jetson/Pictures/m3/A")
    with camera as camera:
        while True :
            # if PLC.read_holding_registers('Y0.11', 1)[0]:
            try:
                _info, _new, old = process(camera)
                if _info:

                    presence =    [ frame.presence for frame in _info ][::-1]
                    orientation = [ frame.orientation for frame in _info ][::-1]
                    print(orientation)
                    enviar_valores_para_clp(100, presence, 20)
                    enviar_valores_para_clp(300, orientation, 20)
                    PLC.write_multiple_coils(CAMERA_TRIGGER_OK, [True])
                    cv2.imwrite('atual.jpg', old)
                    cv2.imwrite('process.jpg', _new)
            except IndexError:
                print("Index Error: modbus connection fail?")
                pass
        #cv2.waitKey(1)
                
#                angle = [ frame.features.angle for frame in _info ]
#                center = [ frame.features.center for frame in _info ]
#                print(angle)
#                print(center)

