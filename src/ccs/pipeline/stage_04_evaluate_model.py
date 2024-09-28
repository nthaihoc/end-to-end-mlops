from ccs import logger
from ccs.components.evaluate_model import EvaluateModel
from ccs.config.configuration import ConfigurationManager

STAGE_NAME = "EVALUATION MODEL"

class EvaluateModelPipeline:
    def __init__ (self):
        pass

    def main(self):
        config = ConfigurationManager()
        evaluate_config = config.setup_evaluate_model()
        evaluate_model = EvaluateModel(config=evaluate_config)
        evaluate_model.evaluation()
        evaluate_model.save_score()
        evaluate_model.log_into_mlflow()
    

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        evaluate_model = EvaluateModelPipeline()
        evaluate_model.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\n x============x")
    except Exception as e:
        logger.exception(e)
        raise e