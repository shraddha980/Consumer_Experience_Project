from Sentiment.entity import config_entity
from Sentiment.entity import artifact_entity
from Sentiment.components import data_ingestion
from Sentiment.components import data_transformation
from Sentiment.logger import logging
from Sentiment.exception import SentimentException
import sys
from Sentiment import utils
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:

    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"Initializing Attributes ")
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
            logging.info(f"Completed Initializing the Attributes")
        except Exception as e:
            raise SentimentException(e,sys)

    def train_model(self, x, y):
        try:
            rf = RandomForestClassifier()
            rf.fit(x,y)
            return rf
        except Exception as e:
            raise SentimentException(e, sys)



    def initiate_model_trainer(self) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"Importing Train and Test Variables")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transform_train_path)
            logging.info(f"Shape of Train array is : {train_arr.shape}")
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transform_test_path)
            logging.info(f"Shape of Test array is : {test_arr.shape}")
            x_train,y_train = train_arr[:,:-1],train_arr[:,-1]
            logging.info(f"Shape of x_train and y_train is {x_train.shape}, {y_train.shape}")
            x_test,y_test = test_arr[:,:-1],test_arr[:,-1]
            logging.info(f"Shape of x_test and y_test is {x_test.shape}, {y_test.shape}")

            logging.info(f"Train the model")
            model = self.train_model(x=x_train, y=y_train)
            logging.info(f"Predicting yhat_train and yhat_test")
            yhat_train = model.predict(x_train)
            logging.info(f"Content of yhat_train is {yhat_train}")
            yhat_test = model.predict(x_test)
            logging.info(f"Content of yhat_test is {yhat_test}")

            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path)
            return model_trainer_artifact

        except Exception as e:
            raise SentimentException(e,sys)





