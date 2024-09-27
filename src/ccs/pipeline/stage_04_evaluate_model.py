from ccs import logger
from ccs.components.evaluate_model import EvaluateModel
from ccs.config.configuration import ConfigurationManager


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