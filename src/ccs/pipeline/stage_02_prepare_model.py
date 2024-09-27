from ccs.config.configuration import ConfigurationManager
from ccs.components.prepare_model import PrepareModel
from ccs import logger


class PrepareModelPipeline:
    def __int__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_model_config = config.setup_model_config()
        prepare_model = PrepareModel(config=prepare_model_config)
        model = prepare_model.base_model()
        model = prepare_model.full_model()

