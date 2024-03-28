from modbus import PLC, Camera, CAMERA_TRIGGER_REG, CAMERA_TRIGGER_OK, OK_RECALIBRATE_REG, MODEL_REGISTER,  enviar_valores_para_clp
from detect import validate, _default_rois, _default_steps, loadRois
import  auto_roi.rois as ROIS
import  auto_roi.remap as mapper
import cv2
import pickle
import numpy as np
SHOW = True
mx, my = pickle.load(open('/home/jetson/Documents/test/Transbordo/mp_xy.pkl', 'rb'))

SEND_CENTERS = False

# B_ref = [(299.8, 198.18), (297.1,5), (19.2, 3.2)]
B_ref = [(288.68, 208.08), (287.78, 25.32), (16.76, 27.49)]
offset_garra = [(0, -32.23), (0, -32.23), (0, -32.23)]
B_ref = np.array(B_ref) + np.array(offset_garra)
#279.38, 194.42
#283.78, 162.19, 
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
                return  r[PROCESS.get_roi_tag()], i 
            return  ROIS.merge_with_old(PROCESS.get_roi(), r[PROCESS.get_roi_tag()]), i

    def FIRST():
        return PROCESS.rois["200"] is None or PROCESS.rois["100"] is None

def process(camera):
    # input("PRESS TO VALIDATE")
    if PROCESS.should_trigger() and not PROCESS.FIRST():
        return PROCESS.validade(camera)
    
    if PROCESS.should_recalibrate():
        r, i = PROCESS.update_roi()
        model = PROCESS.get_model_name()
        if (len(r) == int(model)):
            PROCESS.rois[model] = r

            with open(f"{model}_roi", 'wb') as file:
                pickle.dump(r, file)
            
            print(len(r))
            cv2.imwrite(f"rois_{model}.jpg", i)
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

                    if SEND_CENTERS:
                        A_ref, AREF_OFF, _new = mapper.get_A_ref(mapper.dots_rois, _new)
                        print(A_ref)

                    presence =    [ frame.presence for frame in _info ][::-1]
                    orientation = [ frame.orientation for frame in _info ][::-1]
                    
                    enviar_valores_para_clp(804, presence, 20)
                    enviar_valores_para_clp(300, orientation, 20)
                    # print(presence)


                    cv2.imwrite('atual.jpg', old)
                    cv2.imwrite('process.jpg', _new)
                            
                        
                    if SEND_CENTERS:
                        centers =     [ frame.features[-1].center if frame.presence else (0,0) for frame in _info ]
                        A_ref = mapper.adjust_offset(A_ref, AREF_OFF)
                        offsets = [(frame.roi[0], frame.roi[1]) for frame in _info]
                        centers = mapper.adjust_offset(centers, offsets)
                        centers  = mapper.correlação_planar(centers, A_ref, B_ref)
                        
                        x_array = centers[:, 0][::-1].tolist()
                        y_array = centers[:, 1][::-1].tolist()
                        print(x_array[0], x_array[-1])
                        print(y_array[0], y_array[-1])
                        
                        enviar_valores_para_clp(500, x_array, 20, reg=True)
                        enviar_valores_para_clp(700, y_array, 20, reg=True)

                    PLC.write_multiple_coils(CAMERA_TRIGGER_OK, [True])

            except IndexError as e:
                print("Index Error: modbus connection fail?", e)
                pass

