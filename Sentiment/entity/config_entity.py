import os
import sys
from Sentiment.exception import SentimentException
from Sentiment.logger import logging
from datetime import datetime

FILE_NAME = "IMDB_reviews.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = "transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME = "target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"

class TrainingPipelineConfig:

    def __init__(self):
        try:
            logging.info(f"Start creating TrainingPipeline Directory")
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
            logging.info(f"Completed creating TrainingPipeline Directory")
        except Exception as e:
            raise SentimentException(e,sys)

class DataIngestionConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            logging.info(f"Creating Directories for DataIngestion")
            self.database_name = "NLP_PROJECTS"
            self.collection_name = "SENTIMENT"
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir,"data_ingestion")
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            self.train_file_path = os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size = 0.2
            logging.info(f"Completed creating directories for DataIngestion")
        except Exception as e:
            raise SentimentException(e,sys)

    def to_dict(self) -> dict:
        try:
            logging.info("Start converting dataframe into dictionary")
            return self.__dict__
        except Exception as e:
            raise SentimentException(e,sys)
    logging.info("Completed converting dataframe into dictionary")

class DataTransformationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
            self.transform_object_path = os.path.join(self.data_transformation_dir,"transformed",TRANSFORMER_OBJECT_FILE_NAME)
            self.transform_train_path = os.path.join(self.data_transformation_dir,"transformed",TRAIN_FILE_NAME.replace("csv", "npz"))
            self.transform_test_path = os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME.replace("csv", "npz"))
            self.target_encoder_path = os.path.join(self.data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SentimentException(e,sys)

class DataValidation:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir,"data_validation")
            self.report_file_path = os.path.join(self.data_validation_dir,"report.yaml")
            self.base_file_path = os.path.join("IMDB_reviews.csv")
            self.missing_threshold:float = 0.2
        except Exception as e:
            raise SentimentException(e,sys)

class ModelTrainerConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir,"model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir,"model", MODEL_FILE_NAME)
        self.expected_score = 0.7
        self.over_fitting_threshold = 0.3

class ModelPusherConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir, "model_pusher")
        self.saved_model_dir = os.path.join("saved_models")
        self.pusher_model_dir = os.path.join(self.model_pusher_dir,"saved_models")
        self.pusher_model_path = os.path.join(self.pusher_model_dir, MODEL_FILE_NAME)
        self.pusher_transformer_path = os.path.join(self.pusher_model_dir, TRANSFORMER_OBJECT_FILE_NAME)

class ModelEvaluationConfig:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.change_threshold = 0.01