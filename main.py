from modbus import PLC, Camera, CAMERA_TRIGGER_REG, CAMERA_TRIGGER_OK, OK_RECALIBRATE_REG, MODEL_REGISTER,  enviar_valores_para_clp
from detect import validate, _default_rois, _default_steps, loadRois
import  auto_roi.rois as ROIS
import  auto_roi.remap as mapper
import cv2
import pickle
SHOW = True
mx, my = pickle.load(open('/home/jetson/Documents/test/Transbordo/mp_xy.pkl', 'rb'))

SEND_CENTERS = False

B_ref = [(0,0), (0,100), (100,0)]
class PROCESS:

    rois = {
        "200": loadRois("200_roi",(0,0)),
        "100": loadRois("100_roi",(0,0)),
    }
    
    def camera_read(camera):
        for _ in range(3):
            _info, _frame = camera.read()
            _frame = cv2.remap(_frame, mx, my, cv2.INTER_LINEAR)
        return _info, _frame
    
    
    def should_trigger():
        return PLC.read_coils(CAMERA_TRIGGER_REG, 1)[0]
    
    
    def should_recalibrate():
        return PLC.read_coils(OK_RECALIBRATE_REG, 1)[0]


    def get_model():
        return PLC.read_coils(MODEL_REGISTER, 1)[0]
    
    
    def get_model_name():
        return '200' if PROCESS.get_model() else '100'

    
    def get_roi_tag():
        return 'merge' if PROCESS.get_model() else 'mask'

    
    def get_roi():
        return PROCESS.rois[PROCESS.get_model_name()]

    def validade(camera):
        _info, _frame = PROCESS.camera_read(camera)
        if _info:
            return validate(_frame,  _default_steps, PROCESS.get_roi(), (10,10), 20)

    def update_roi():
        _info, _frame = PROCESS.camera_read(camera)
        if _info:
            r, i = ROIS.new(_img=_frame, model=PROCESS.get_model_name())
            if PROCESS.FIRST():
                return r[PROCESS.get_roi_tag()], i 
            return ROIS.merge_with_old(PROCESS.get_roi(), r[PROCESS.get_roi_tag()]), i

    def FIRST():
        return PROCESS.rois["200"] is None or PROCESS.rois["100"] is None

def process(camera):
    if PROCESS.should_trigger() and not PROCESS.FIRST():
        return PROCESS.validade(camera)
    
    if PROCESS.should_recalibrate():
        r, i = PROCESS.update_roi()
        PROCESS.rois[PROCESS.get_model_name()] = r

        with open(f"{PROCESS.get_model_name()}_roi", 'wb') as file:
            pickle.dump(r, file)
        
        cv2.imwrite(f"rois_{PROCESS.get_model_name()}.jpg", i)
    PLC.write_multiple_coils(CAMERA_TRIGGER_REG,[False,False])
    return None, None, None


if __name__ == '__main__':
    _default_steps = _default_steps()
    camera = Camera(False)#, "./calibration")
    with camera as camera:
        while cv2.waitKey(1) != 27:
            try:
                _info, _new, old = process(camera)
                if _info:

                    presence =    [ frame.presence for frame in _info ][::-1]
                    orientation = [ frame.orientation for frame in _info ][::-1]
                    
                    enviar_valores_para_clp(100, presence, 20)
                    enviar_valores_para_clp(300, orientation, 20)

                    if SEND_CENTERS:
                        centers =     [ frame.features[-1].center if frame.presence else (0,0) for frame in _info ]

                        A_ref = mapper.get_A_ref(mapper.dots_rois, _new)
                        centers  = mapper.correlação_planar(centers, A_ref, B_ref)
                        
                        x_array = centers[:, 0]
                        y_array = centers[:, 1]

                        enviar_valores_para_clp(500, x_array, 20)
                        enviar_valores_para_clp(700, y_array, 20)

                    PLC.write_multiple_coils(CAMERA_TRIGGER_OK, [True])
                    cv2.imwrite('atual.jpg', old)
                    cv2.imwrite('process.jpg', _new)

            except IndexError:
                print("Index Error: modbus connection fail?")
                pass

