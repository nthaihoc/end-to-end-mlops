from ccs.config.configuration import ConfigurationManager
from ccs.components.prepare_model import PrepareModel
from ccs import logger


STAGE_NAME = "PREPARE MODEL"

class PrepareModelPipeline:
    def __int__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_model_config = config.setup_model_config()
        prepare_model = PrepareModel(config=prepare_model_config)
        model = prepare_model.base_model()
        model = prepare_model.full_model()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        prepare_model = PrepareModelPipeline()
        prepare_model.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n x============x")
    except Exception as e:
        logger.exception(e)
        raise e
