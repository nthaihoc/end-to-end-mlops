import os
from ccs import logger
from ccs.config.configuration import PrepareModelConfig
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Model

class PrepareModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config
    

    def base_model(self):
        self.model = tf.keras.applications.MobileNetV2(
            input_shape = self.config.image_size,
            include_top = self.config.include_top,
            weights     = self.config.weights
        )

    @staticmethod
    def pre_trained_model(model, output, learning_rate, 
                          beta_1, beta_2, weight_decay):

        model.trainable=False
        x = tf.keras.layers.Conv2D(64, (3, 3),
                                   activation="relu",
                                   padding="same") (model.output)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                         padding="same") (x)
        x = tf.keras.layers.BatchNormalization() (x)
        x = tf.keras.layers.Conv2D(128, (3, 3),
                                   activation="relu",
                                   padding="same",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001)) (x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                         padding="same") (x)
        x = tf.keras.layers.BatchNormalization() (x)

        x = tf.keras.layers.Flatten() (x)
        x = tf.keras.layers.Dropout(0.3) (x)
        x = tf.keras.layers.Dense(100, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)) (x)
        x = tf.keras.layers.Dropout(0.3) (x)
        x = tf.keras.layers.Dense(output, activation="softmax") (x)

        model = Model(inputs=model.input, outputs=x)

        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                         beta_1=beta_1,
                                                         beta_2=beta_2,
                                                         weight_decay=weight_decay),
                                                         metrics=["accuracy"])
        return model
    
    def full_model(self):
        self.full_model = self.pre_trained_model(
            model = self.model,
            output= self.config.classes,
            learning_rate=self.config.learning_rate,
            beta_1=self.config.beta_1,
            beta_2=self.config.beta_2,
            weight_decay=self.config.decay
        )

        self.save_model(self.config.base_model_dir, self.full_model)

    @staticmethod
    def save_model(path: Path, model=tf.keras.Model):
        model.save(path)

