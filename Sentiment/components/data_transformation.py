import pandas as pd
import numpy as np
import sys
from Sentiment.logger import logging
from Sentiment.exception import SentimentException
from Sentiment.entity import config_entity
from Sentiment.entity import artifact_entity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
from Sentiment import utils
from sklearn.pipeline import Pipeline

class DataTransformation:

    def __init__(self, data_transformation_config:config_entity.DataTransformationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"Creating Data Transformation Attributes")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise SentimentException(e, sys)

    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        try:
            cv = CountVectorizer()
            return cv
        except Exception as e:
            raise SentimentException(e,sys)

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"Shape of train_df: , {train_df.shape}")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"shape of test_df: , {test_df.shape}")

            train_input_feature = train_df.iloc[:,-2]
            logging.info(f"shape of train_input_feature is:  {train_input_feature.shape}")
            logging.info(f"Content of train_input_feature: {train_input_feature.head(10)}")

            train_target_feature = train_df.iloc[:,-1]
            logging.info(f"shape of train_target_feature {train_target_feature.shape}")
            logging.info(f"Content of train_target_feature {train_target_feature.head(10)}")

            test_input_feature = test_df.iloc[:,-2]
            logging.info(f"shape of test_input_feature: {test_input_feature.shape}")
            test_target_feature = test_df.iloc[:,-1]
            logging.info(f"Shape of test_target_feature:  {test_target_feature.shape}")

            label_encoder = LabelEncoder()
            train_target_feature = label_encoder.fit_transform(train_target_feature)
            logging.info(f"Content after applying label encoder on train target: {train_target_feature}")
            test_target_feature = label_encoder.fit_transform(test_target_feature)
            logging.info(f"Content after applying label encoder on test target: {test_target_feature}")

            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(train_input_feature)
            logging.info(f"Applying CV Transformation")

            train_input_feature = transformation_pipeline.transform(train_input_feature).toarray()
            logging.info(f"Content after applying CV transform : {train_input_feature}")
            train_input_feature = train_input_feature.astype(np.uint8)
            logging.info(f"Content after applying cv transform: {train_input_feature}")

            cv = CountVectorizer()
            test_input_feature = transformation_pipeline.transform(test_input_feature).toarray()
            test_input_feature = test_input_feature.astype(np.uint8)
            logging.info(f"Content after applying cv transform: {test_input_feature}")

            #train_target_feature = train_target_feature.squeeze()
            #test_target_feature = test_target_feature.squeeze()

            train_arr = np.c_[train_input_feature,train_target_feature]
            test_arr = np.c_[test_input_feature,test_target_feature]

            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_train_path,
                                       array=train_arr)
            utils.save_numpy_array_data(file_path=self.data_transformation_config.transform_test_path,
                                       array=test_arr)
            utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)

            utils.save_object(file_path=self.data_transformation_config.target_encoder_path, obj = label_encoder)

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transform_train_path=self.data_transformation_config.transform_train_path,
                transform_test_path=self.data_transformation_config.transform_test_path,
                target_encoder_path=self.data_transformation_config.target_encoder_path,
                transform_object_path=self.data_transformation_config.transform_object_path
                )

            return data_transformation_artifact
        except Exception as e:
            raise SentimentException(e,sys)



