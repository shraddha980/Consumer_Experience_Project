import os
import sys
import numpy as np
from Sentiment.predictor import ModelResolver
import pandas as pd
from Sentiment.logger import logging
from Sentiment.exception import SentimentException
from Sentiment.entity import artifact_entity
from Sentiment.entity import config_entity
from Sentiment import utils
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from Sentiment.components import data_ingestion,data_transformation,model_pusher,model_trainer
from Sentiment.entity.artifact_entity import DataIngestionArtifact,DataTransformationArtifact,ModelPusherArtifact,ModelTrainerArtifact



class ModelEvaluation:

    def __init__(self,
                 model_evaluation_config:config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifact_entity.ModelTrainerArtifact
                 ):
        try:
            logging.info(f"Initiating Model Evaluation ")
            self.model_evaluation_config:model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
            logging.info(f"Completed initiating Model Evaluation")

        except Exception as e:
            raise SentimentException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info(f"If saved model folder has model then we will compare"
                         "which model is the best trained, current model or model"
                         "from the saved folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True)
                return model_eval_artifact


            logging.info(f"Importing Saved Model Paths from respective locations")
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()
            transformer_path= self.model_resolver.get_latest_transform_path()

            logging.info(f" saving Previously Trained Models in model and target_encoder")
            transformer = utils.load_object(file_path=transformer_path)
            model = utils.load_object(file_path=model_path)
            target_encoder_path = utils.load_object(file_path=target_encoder_path)

            logging.info(f"Importing Currently Trained Model Paths")
            current_model= utils.load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder= utils.load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            logging.info(f"Taking Test File Path")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info(f"Shape of test df is : {test_df.shape}")


            pattern = re.compile('<.*?>')

            def clean_html(text):
                clean_text = re.sub(pattern,"", text)
                return clean_text

            test_df['review'] = test_df['review'].apply(clean_html)
            test_df['review'] = test_df['review'].apply(lambda x: x.lower())
            sw_list = stopwords.words('english')
            test_df['review'] = test_df['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(
                lambda x: " ".join(x))

            test_input_df = test_df.iloc[:, -2]
            logging.info(f"test input df is {test_input_df}")
            test_target_df = test_df.iloc[:, -1]
            logging.info(f"test_target_df is {test_target_df}")
            y_true = test_target_df

            #transformed_test_input_df = list(transformer.feature_names_in)
            transformed_input_array = transformer.transform(test_input_df)
            transformed_test_input_df  = transformed_input_array.astype(np.uint8)
            logging.info(f"transformed_test_input_df is {transformed_test_input_df}")

            label_encoder = LabelEncoder()
            test_target_df = label_encoder.fit_transform(test_target_df)

            y_prediction = model.predict(transformed_test_input_df)
            print(f"prediction using Trained Model: {y_prediction}")
            logging.info(f"prediction using Trained Model: {y_prediction}")
            y_prediction_score = f1_score(test_target_df,y_prediction)
            print(f"F1 score of Previous Model is : {y_prediction_score}")
            logging.info(f"F1 score of Previous Model is : {y_prediction_score}")

            y_prediction_current = current_model.predict(transformed_test_input_df)
            print(f"Prediction using current Model: {y_prediction_current}")
            logging.info(f"prediction using Trained Model: {y_prediction_current}")
            y_prediction_score_current = f1_score(test_target_df,y_prediction_current)
            print(f"F1 score of Current Model is : {y_prediction_score_current}")
            logging.info(f"F1 score of Current Model is : {y_prediction_score_current}")

            if y_prediction_score_current <= y_prediction_score:
                logging.info(f"Current trained model is not better than Previous Trained Model")
                raise Exception(f"Current trained model is not better than Previous Trained Model")

            model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=y_prediction_score_current-y_prediction_score)
            return model_evaluation_artifact
        except Exception as e:
            raise SentimentException(e,sys)