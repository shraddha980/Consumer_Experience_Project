from Sentiment.logger import logging
from Sentiment.exception import SentimentException
import sys
import os
from Sentiment.components import data_transformation
from Sentiment.components import model_trainer
from Sentiment.entity import artifact_entity
from Sentiment.entity import config_entity
from Sentiment.entity.config_entity import ModelPusherConfig
from Sentiment import utils
from Sentiment import predictor
from Sentiment.predictor import ModelResolver
from Sentiment.utils import save_object,load_object
from Sentiment.entity.artifact_entity import ModelTrainerArtifact, ModelPusherArtifact,DataTransformationArtifact,DataIngestionArtifact

class ModelPusher:

    def __init__(self,data_transformation_artifact:DataTransformationArtifact,
                      model_pusher_config:ModelPusherConfig,
                      model_trainer_artifact:ModelTrainerArtifact
                      ):
        try:
            logging.info(f"Initiating Model Pusher Arguments")
            self.data_transformation_artifact= data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_pusher_config = model_pusher_config
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise SentimentException(e,sys)

    def initiate_model_pusher(self) -> artifact_entity.ModelPusherArtifact:
        try:
            logging.info(f"Download Transformer Artifact and Model Trainer Artifact")
            transformer = utils.load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = utils.load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder = utils.load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            logging.info(f"Save models in Model Pusher Directory")
            save_object(file_path=self.model_pusher_config.pusher_transformer_path, obj=transformer)
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj = model)
            save_object(file_path=self.data_transformation_artifact.target_encoder_path, obj = target_encoder)

            logging.info(f"Saving model in saved Model Directory")
            transformer_path = self.model_resolver.get_saved_transformer_path()
            model_path = self.model_resolver.get_saved_model_path()
            target_encoder_path = self.model_resolver.get_saved_target_encoder_path()

            save_object(file_path=model_path, obj=model)
            save_object(file_path=target_encoder_path, obj=target_encoder)
            save_object(file_path=transformer_path, obj=transformer)

            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                                                        saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Pusher Model Artifact : {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise SentimentException
