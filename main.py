from Sentiment.logger import logging
import sys
from Sentiment.exception import SentimentException
from Sentiment.entity import config_entity
from Sentiment.entity import artifact_entity
from Sentiment.components.data_ingestion import DataIngestion
from Sentiment.components.data_transformation import DataTransformation
from Sentiment.components.model_trainer import ModelTrainer
from Sentiment.components.model_pusher import ModelPusher
from Sentiment.components.model_evaluation import ModelEvaluation
from Sentiment.predictor import ModelResolver
from Sentiment.entity.config_entity import ModelEvaluationConfig
from Sentiment.entity.artifact_entity import DataTransformationArtifact,ModelPusherArtifact,ModelTrainerArtifact,DataIngestionArtifact

from Sentiment import utils

if __name__ == "__main__":
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config,
                                                 data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact=data_transformation.initiate_data_transformation()

        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config,
                                   data_transformation_artifact=data_transformation_artifact,
                                   model_trainer_artifact=model_trainer_artifact)
        model_pusher_artifact = model_pusher.initiate_model_pusher()

        model_eval_config = config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
        model_eval = ModelEvaluation(model_evaluation_config=model_eval_config,
                                     data_transformation_artifact=data_transformation_artifact,
                                     data_ingestion_artifact=data_ingestion_artifact,
                                     model_trainer_artifact=model_trainer_artifact)
        model_evaluation_artifact = model_eval.initiate_model_evaluation()

    except Exception as e:
        print(e)
