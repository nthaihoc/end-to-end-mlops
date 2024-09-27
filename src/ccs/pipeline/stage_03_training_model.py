from ccs import logger
from ccs.components.model_training import TrainingModel
from ccs.config.configuration import ConfigurationManager


class TrainModelPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_config = config.setup_train_model()
        model_train = TrainingModel(config=prepare_config)
        model_train.train_model()
        