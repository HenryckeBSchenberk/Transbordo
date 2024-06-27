import numpy as np
from datetime import datetime
from os import makedirs
from tensorflow.keras.models import (
        load_model
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from codetiming import Timer
class ModelManager:
    def __init__(self, name, model=None, batch_size=200 ):
        self.name=name
        if model is not None:
            self.model = model

    @Timer(name="IdentifyProcess",  text="{name} demorou: {:.4f} segundos")
    def test(self, frames):
        try:
            return self.model.predict(np.array(frames), batch_size=200)
        except Exception:
            raise ValueError("O modelo não consegue processar esse formato de informação.")

    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, value:str): 
        if value.endswith((".h5", ".keras")): 
            self.__model = load_model(value)
            print(self.__model.summary())
            return
        raise TypeError("Sported Models: 'h5, keras'")
    
    def createTrainGen(TRAIN_DIR='/app/dataset/cnn/train', batch_size=32, target_size=(128, 128, 3)):
        datagen = ImageDataGenerator(
            brightness_range=[0.4,1.5],
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
        )
        return datagen.flow_from_directory(TRAIN_DIR,
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            target_size=target_size[:2],
                                            follow_links=True)

    def createValidateGen(VALIDATION_DIR='/app/dataset/cnn/test', batch_size=32, target_size=(128, 128, 3)):
        validation_datagen = ImageDataGenerator()
        return validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    target_size=target_size[:2],
                                                    follow_links=True)

    def trainModel(model, train_generator, validation_generator, check_point_path='/app/dataset/models'):
        # Create directories if they don't exist
        check_point_path = check_point_path.rstrip('/')  # Remove trailing slash if exists
        date_str = datetime.now().strftime('%d-%m-%Y-_%H-%M-%S')
        checkpoint_directory = f'{check_point_path}/{date_str}'
        makedirs(checkpoint_directory, exist_ok=True)


        fname = "weights-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.keras"
        callbacks = [
            # EarlyStopping(monitor='val_loss', patience=patience, mode="min"),
            ModelCheckpoint(
                filepath=f'{checkpoint_directory}/{fname}',
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
                initial_value_threshold=0.1
            )
        ]

        history = model.fit(
                train_generator, epochs=50,
                validation_data=validation_generator,
                callbacks=callbacks
        )
        
        return model, history, checkpoint_directory