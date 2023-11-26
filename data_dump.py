#importing libraries
import pandas as pd
import json
import pymongo

#connecting to MongoDB client
client = pymongo.MongoClient("mongodb+srv://nlpprojects809:Crt9VnDR6LTzjhqm@cluster0.wuw5etg.mongodb.net/?retryWrites=true&w=majority")

#reading dataframe from csv file
df = pd.read_csv("E:\SentimentAnalysis\IMDB_reviews.csv")
df = df.iloc[0:1000]
#converting dataframe to json format
json_record = list(json.loads(df.T.to_json()).values())

#inserting json records in MongoDB Database with appropriate Names
client["NLP_PROJECTS"]["SENTIMENT"].insert_many(json_record)


