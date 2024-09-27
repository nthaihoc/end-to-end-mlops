from pathlib import Path
import os
import keras
import tensorflow as tf
from PIL import Image
from ccs.config.configuration import EvaluateModelConfig
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ccs.utils.common import save_json
import mlflow
import mlflow.keras
from urllib.parse import urlparse

class EvaluateModel:
    def __init__ (self, config: EvaluateModelConfig):
        self.config = config

    
    
    def evaluation(self):
        self.model = tf.keras.models.load_model(self.config.model_trained_dir)

        X_test, Y_test = data_processing(path_root=self.config.test_data_path,
                                         path_chil=self.config.foler_name,
                                         path_chil_label=self.config.label_name)
        x_test = img_size(X_test[:100])

        y_test = encoder(Y_test[:100])

        preds = self.model.predict(x_test)

        y_preds = []
        for i in preds:
            y_preds.append(np.argmax(i))

        self.acc = accuracy_score(y_test, y_preds)
        self.pre = precision_score(y_test, y_preds, average="weighted")
        self.recall = recall_score(y_test, y_preds, average="weighted")
        self.f1_score = f1_score(y_test, y_preds  , average="weighted")

    
    def save_score(self):
        scores = {"accuracy" : self.acc,
                  "precision": self.pre,
                  "recall"   : self.recall,
                  "f1"       : self.f1_score}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_param)
            mlflow.log_metrics(
                {"accuracy" : self.acc, 
                 "precision": self.pre,
                 "recall"   : self.recall, 
                 "f1"       : self.f1_score}
            )
            if tracking_url_type_store != "file":


                mlflow.keras.log_model(self.model, "model", registered_model_name="MobiNetV2")
            else:
                mlflow.keras.log_model(self.model, "model")

def data_processing(path_root: Path, 
                    path_chil: list, 
                    path_chil_label: list):
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
    return X, Y

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