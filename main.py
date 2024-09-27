from ccs import logger
from ccs.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from ccs.pipeline.stage_02_prepare_model import PrepareModelPipeline
from ccs.pipeline.stage_03_training_model import TrainModelPipeline
from ccs.pipeline.stage_04_evaluate_model import EvaluateModelPipeline
from PIL import Image
import os
import numpy as np
from pathlib import Path

# STAGE_NAME = "Data Ingestion Stage"
# try:
#     logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
#     data_ingestion = DataIngestionPipeline()
#     data_ingestion.main()
#     logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Prepare Model"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    prepare_model = PrepareModelPipeline()
    prepare_model.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Training Model"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    train_model = TrainModelPipeline()
    train_model.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluate Model"
try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    evaluation_model = EvaluateModelPipeline()
    evaluation_model.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
except Exception as e:
    logger.exception(e)
    raise e


