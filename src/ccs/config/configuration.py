from ccs.constants import *
from ccs.utils.common import read_yaml, create_directories
from ccs.entity.config_entity import (DataIngestionConfig,
                                      PrepareModelConfig,
                                      TrainingModelConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        #create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        #create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            data_URL=config.data_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def setup_model_config(self) -> PrepareModelConfig:
        
        config = self.config.pre_model

        create_directories([config.root_dir])

        _set_up_model_config = PrepareModelConfig(
            root_dir       = config.root_dir,
            base_model_dir = config.base_model_dir,
            image_size     = self.params.IMAGE_SIZE,
            learning_rate  = self.params.LEARNING_RATE,
            include_top    = self.params.INCLUDE_TOP,
            weights        = self.params.WEIGHTS,
            classes        = self.params.CLASSES,
            beta_1         = self.params.BETA_1,
            beta_2         = self.params.BETA_2,
            decay          = self.params.DECAY
        )

        return _set_up_model_config
    
    def setup_train_model(self) -> TrainingModelConfig:

        config = self.config.pre_model

        _setup_train_model = TrainingModelConfig(
            model_trained_dir = config.base_model_dir,
            epochs            = self.params.EPOCHS,
            list_folder_name  = self.params.LIST_FOLDER_NAME,
            list_label_name   = self.params.LIST_LABEL_NAME,
            root_dir          = config.data_dir,
            batch_size        = self.params.BATCH_SIZE
         )

        return _setup_train_model
