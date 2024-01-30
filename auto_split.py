from split_data import main
import os

# classes_folder_name = ['2','3']
classes_model = ['A','B']
groups = ['test', 'train']


fpp = "C:/Users/Henrycke/Documents/GitHub/Transbordo/"



prefix_origin = fpp#+"database/config/db"
prefix_output= fpp#+"database/dataset"

roi_path = "C:/Users/Henrycke/Documents/GitHub/Transbordo/200_roi"

# for folder in classes_folder_name:
for model in classes_model:
    for group in groups:
        inp = f"{prefix_origin}/{model}/{group}"
        out = f"{prefix_output}/{model}/{group}"
        if not os.path.exists(out):
            os.makedirs(out)
        main([
            "-LF", inp,
            "-AR", fpp,
            "-ND",
            "-SF", out,
            "-LR", roi_path
        ])

# python .\split_data.py -LF "./database/config/db/1/nok/test" -AR "./"  -ND -SF "./database/dataset/1/nok/test" -LR "./new_rois"