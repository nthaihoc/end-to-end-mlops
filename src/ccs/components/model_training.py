import os 
import tensorflow as tf
from ccs.config.configuration import TrainingModelConfig
from ccs import logger
from pathlib import Path
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class TrainingModel:
    def __init__ (self, config: TrainingModelConfig):
        self.config = config

    def train_model(self):
        X_full, Y_full = data_processing(self.config.root_dir,
                                         self.config.list_folder_name,
                                         self.config.list_label_name)
        X_train, X_dev, X_test = X_full
        Y_train, Y_dev, Y_test = Y_full
        
        logger.info("Resize Image and Convert Matrix")
        x_train, x_dev, x_test = (img_size(X_train[:100]),
                                  img_size(X_dev[:100]),
                                  img_size(X_test[:100]))
        logger.info("Resize Image and Convert Matrix Successfully")

        logger.info("OneHotEncoder Label")
        y_train, y_dev, y_test = (encoder(Y_train[:100]),
                                  encoder(Y_dev[:100]),
                                  encoder(Y_test[:100]))
        logger.info("OneHotEncoder Label Successfully")

        logger.info("Data Augmentation Processing")
        train_datagen = ImageDataGenerator(rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

        dev_datagen = ImageDataGenerator()

        train =  train_datagen.flow(x_train, y_train, batch_size=self.config.batch_size)
    
        dev = dev_datagen.flow(x_dev ,y_dev, batch_size=self.config.batch_size)
        logger.info("Data Augmentation Successfully")

        self.model = tf.keras.models.load_model(self.config.model_train_dir)
       
        self.model.fit(x=train,
                       validation_data=dev,
                       epochs=self.config.epochs,
                       shuffle=True)
    
        self.save_model(self.config.model_trained_dir, self.model)

    @staticmethod
    def save_model(path: Path, model=tf.keras.Model):
        model.save(path)
        
def data_processing(path_root: Path, path_chil: list, path_chil_label: list):
    X_full = []
    Y_full = []
    for i in path_chil:
        X = []
        Y = []
        path = os.path.join(path_root, i)
        for j in path_chil_label:
            path_label = os.path.join(path, j)
            for filename in os.listdir(path_label):
                file_img = os.path.join(path_label, filename)
                X.append(file_img)
                Y.append(j)

        X_full.append(X)
        Y_full.append(Y)

    return X_full, Y_full


def img_size(x: list):
    X = []
    for i in x:
        img = Image.open(i)
        if img.mode == 'RGBA':
            img = img.convert('RGB') 
        img_resize = img.resize((224, 224))
        X.append(img_resize)
    return np.array(X).astype("float32") / 255.0


def encoder(y: list):
    Y = []
    for i in y:
        if (i == "ASC_H"):
            Y.append(0)
        if (i == "ASC_US"):
            Y.append(1)
        if (i == "HSIL"):
            Y.append(2)
        if (i == "LSIL"):
            Y.append(3)
        if (i == "SCC"):
            Y.append(4)
    return np.array(Y).astype("float32")

