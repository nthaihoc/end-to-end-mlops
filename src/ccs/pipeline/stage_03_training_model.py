from ccs import logger
from ccs.components.model_training import TrainingModel
from ccs.config.configuration import ConfigurationManager

STAGE_NAME = "TRAINING MODEL"

class TrainModelPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_config = config.setup_train_model()
        model_train = TrainingModel(config=prepare_config)
        model_train.train_model()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        train_model = TrainModelPipeline()
        train_model.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n x============x")
    except Exception as e:
        logger.exception(e)
        raise e
        