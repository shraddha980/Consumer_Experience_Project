import pandas as pd
from Sentiment.exception import SentimentException
from Sentiment.logger import logging
import os,sys
import numpy as np
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from Sentiment.entity import config_entity
from Sentiment.entity import artifact_entity
from Sentiment import utils
from nltk.corpus import stopwords
import re


class DataIngestion:

    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SentimentException(e,sys)

    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f"Exporting collection data as DataFrame")
            df = pd.read_csv("IMDB_reviews.csv")
            df = df.iloc[0:1000]
            logging.info(f"Shape of DataFrame is: , {df.shape}")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)
            df.drop_duplicates(inplace=True)
            pattern = re.compile('<.*?>')

            def clean_html(text):
                clean_text = re.sub(pattern,"", text)
                return clean_text

            pattern = re.compile('<.*?>')
            df['review'] = df['review'].apply(clean_html)
            df['review'] = df['review'].apply(lambda x: x.lower())
            sw_list = stopwords.words('english')
            df['review'] = df['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(
                lambda x: " ".join(x))
            train_df,test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size,random_state=1)
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)
            train_df.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path)
            return data_ingestion_artifact

        except Exception as e:
            raise SentimentException(e,sys)