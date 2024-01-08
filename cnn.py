import keras
import tensorflow as tf
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from os import makedirs
from datetime import datetime
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *

def CreateModel(check_point_path='./model/models', TRAIN_DIR='./model/train/', VALIDATION_DIR='./model/test/', batch_size=250, target_size=(128, 128, 3), patience=5):

    # Create directories if they don't exist
    check_point_path = check_point_path.rstrip('/')  # Remove trailing slash if exists
    date_str = datetime.now().strftime('%d-%m-%Y-_%H-%M-%S')
    checkpoint_directory = f'{check_point_path}/{date_str}'
    makedirs(checkpoint_directory, exist_ok=True)


    fname = "weights-{epoch:03d}-{val_loss:.4f}-{val_acc:.4f}.h5"
    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=patience, mode="min"),
        ModelCheckpoint(
            filepath=f'{checkpoint_directory}/{fname}',
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
            initial_value_threshold=0.06
        )
    ]

    model = keras.Sequential([
        #InputLayer
        Conv2D(
            filters=32, kernel_size=(3,3), padding='same',
            activation='relu', input_shape=(128, 128, 3)
        ),
        #Convolutional
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.2),
        #==================================
        Conv2D(64, (3,3), padding='same', activation='relu'),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.2),
        #==================================
        Conv2D(128, (3,3), padding='same', activation='relu'),
        Conv2D(128, (3,3), activation='relu'),
        Activation('relu'),
        MaxPooling2D((2,2)),
        Dropout(0.2),
        #==================================
        Conv2D(512, (5,5), padding='same', activation='relu'),
        Conv2D(512, (5,5), activation='relu'),
        MaxPooling2D((4,4)),
        Dropout(0.2),
        #==================================
        #### Fully-Connected Layer ####
        #==================================
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')
    ])

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=target_size),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(256/2, activation='relu'),
    #     tf.keras.layers.Dense(256/2, activation='relu'),
    #     tf.keras.layers.Dense(256/2, activation='relu'),
    #     tf.keras.layers.Dense(256/2, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])

    model.compile(optimizer=RMSprop(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['acc'])

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.1
    )
    train_generator = datagen.flow_from_directory(TRAIN_DIR,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        target_size=target_size[:2])


    validation_datagen = ImageDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                                batch_size=batch_size,
                                                                class_mode='binary',
                                                                target_size=target_size[:2])
    history = model.fit(
            train_generator, epochs=1000,
            validation_data=validation_generator,
            callbacks=callbacks
    )
    
    return model, history

import argparse
import numpy as np
from split_data import loadFrames

def main():
    parser = argparse.ArgumentParser(description='Process images and ROIs.')
    parser.add_argument('-LM', '--LoadModel', type=str, help='Path for model.h5')
    parser.add_argument('-CM', '--CreateModel', type=str, help='Create model')
    parser.add_argument('-LF', '--LoadFrames', type=str, help='Load frames from a directory')
    
    parser.add_argument('-CKP', '--checkpoint_path', type=str, help='path to save model.h5 while creating a new one', default='./model/models/')
    parser.add_argument('-P', '--patience', type=int, help='Hits to stop epoch generation. eg: 5', default=30)
    parser.add_argument('-VD', '--validation_dir', type=str, help='path to dir with validation images. (classes should be on subfolder)', default='./model/test')
    parser.add_argument('-TD', '--training_dir', type=str, help='path to dir with training images. (classes should be on subfolder)', default='./model/train')
    parser.add_argument('-BS', '--batch_size', type=int, help='batch_size for images eg:250', default=250)
    parser.add_argument('-TS', '--target_size', type=tuple, help='shape/size og images eg: (128, 128, 3)', default=(128,128,3))


    
    args = parser.parse_args()

    if args.CreateModel:
        model, history = CreateModel(
            args.checkpoint_path,
            args.training_dir,
            args.validation_dir,
            args.batch_size,
            args.target_size,
            args.patience
        )
        print(model.summary())
    
    if args.LoadModel:
        model = keras.models.load_model(args.LoadModel)
        if args.LoadFrames:
            # import cv2
            # img = cv2.imread('./model/test/ok/8604ba839dca11eea2d9e0d55ef67131.jpg')
            frames = loadFrames(args.LoadFrames)
            result = model.predict(np.array(frames))

            nok = len(list(filter(lambda x: x<0.5, result)))
            ok = len(list(filter(lambda x: x>=0.5, result)))
            sz = len(result)
            print(
                round(ok/sz, 3),f"% [{ok}/sz] - ok \n",
                round(nok/sz, 3),f"% [{nok}/sz] - nok"
            )
        else:
            print(model.summary())

if __name__ == '__main__':
    main()