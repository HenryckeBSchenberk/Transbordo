import tensorflow as tf
from threading import Thread
import numpy as np
import tensorflow.keras as keras
from interface.prediction_interface import ModelManager
from operator import methodcaller
from base_service import service
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from cv2 import imread

def update(zip_content):
    return zip_content[0].update('prediction', zip_content[1])

def callback(command, manager):
    action = command.get('action')
    match action:
        case 'load_model' :
            try:
                model_path = command.get('model_path')
                manager.model = model_path
                payload = {"msg":"Modelo carregado com sucesso", 'to_output':False}
            except (TypeError) as e:
                payload = {"msg":"Falha ao carregar o modelo", "err": e, 'to_output':False}
            except Exception as e:
                payload = {"msg": "Erro inesperado durante o carregamento do modelo", "err": e, 'to_output':False}
        case 'predict' :
            try: 
                value = command.get('value')
                result =manager.test(np.array([filtred.normalized_data for filtred in value]))

                payload = {'action':'interpret','value':list(map(update, zip(value, result))), 'to_output':True}
            except ValueError as e:
                payload = {'msg': "Falha ao predizer resultados", 'err':e, 'to_output':False}
        case 'train':
            train_data=ModelManager.createTrainGen()
            validate_data=ModelManager.createValidateGen()
            new_model, history, ckp_dir = ModelManager.trainModel(manager.model, train_data, validate_data)
            df = pd.DataFrame(history.history).rename_axis('epoch').reset_index().melt(id_vars=['epoch'])
            fig, axes = plt.subplots(1,2, figsize=(18,6))
            for ax, mtr in zip(axes.flat, ['loss', 'accuracy']):
                ax.set_title(f'{mtr.title()} Plot')
                dfTmp = df[df['variable'].str.contains(mtr)]
                sns.lineplot(data=dfTmp, x='epoch', y='value', hue='variable', ax=ax)
            fig.tight_layout()
            plt.savefig(f"{ckp_dir}/history.jpg")
            with open(f"{ckp_dir}/history.pkl", 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            return {}
        case _:
            payload = {'msg':'ação desconhecida', 'command':command, 'to_output':False}
    return payload

if __name__ == "__main__":
    import argparse
    

    parser = argparse.ArgumentParser(description="Client script for capturing images and sending commands to the model server.")
    parser.add_argument('--service_host', type=str, required=False, help="(HOST) where the this service are.", default="")
    parser.add_argument('--service_port', type=str, required=False, help="(PORT) where the this service are.", default="5000")
    parser.add_argument('--output_host', type=str, required=False, help="(HOST) where the information goes.", default="tensorflow")
    parser.add_argument('--output_port', type=str, required=False, help="(PORT) where the information goes.", default="6000")
    
    args = parser.parse_args()
    model_mg = ModelManager("identify", 'weights-009-0.0073-0.9993.keras')
    model_mg.test(np.array(np.zeros((200, 128, 128, 3))))
    callback_args = (model_mg, )
    server_process = Thread(target=service, args=((args.service_host, int(args.service_port)), (args.output_host, int(args.output_port)), callback, *callback_args))

    print(f"{__file__} started.")
    server_process.start()
    server_process.join()
    print(f"{__file__} stopped.")