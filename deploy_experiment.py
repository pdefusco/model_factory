from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import os

#Picking the most promising experiment

experiments_summary_df = pd.read_csv("experiments_summary/experiments_summary.csv")

experiments_summary_df.sort_values("BEST_SCORE", ascending=False).head()

#TODO: save model grid in the file so you use the actual best model hyperparams

clf = experiments_summary_df.loc[experiments_summary_df['BEST_SCORE'].idxmax()]['CLASSIFIER'] + '()'

print("Retraining {}".format(best_model))

#Retraining the model with fixed hyperparameters

spark = SparkSession\
    .builder\
    .appName("DeployExperiment")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://demo-aws-1/")\
    .config("spark.hadoop.yarn.resourcemanager.principal",os.getenv("HADOOP_USER_NAME"))\
    .config("spark.executor.instances", 2)\
    .config("spark.executor.cores", 2)\
    .getOrCreate()
    
#To do: take in new data rather than old table

table_name = "default.historical_customer_interactions"

df = spark.sql("select * from {}".format(table_name))

df = df.toPandas()

df = pd.get_dummies(df, columns=["channel", "offer"], drop_first=True)
df = df.drop(columns=['zip_code'])

y = df['conversion']
X = df.drop(columns=['conversion'])

#Redo a train-test split or just retrain on the entire dataset?
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = eval(experiments_summary_df.loc[experiments_summary_df['BEST_SCORE'].idxmax()]['CLASSIFIER'] + '()')
clf.fit(X, y)

#saving the model

#TODO: store model in table or S3

import pickle
from joblib import dump, load

s = pickle.dumps(clf)

dump(clf, 'models/challengers/clf.joblib') 

#Deploy the Challenger model - prepare yml:

from cmlbootstrap import CMLBootstrap
import datetime
import os, time

HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = "uuc48l0gm0r3n2mib27voxazoos65em0"#os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT") 

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

# Create the YAML file for the model lineage
yaml_text = open("lineage.yml","r")
yaml_read = yaml_text.read()

challenger_yaml = '''"Challenger {}":
  hive_table_qualified_names:
    - "{}@cm"
  metadata:
    deployment: "this model was deployed programmatically"'''.format(run_time_suffix, table_name)

yaml_out = yaml_read + challenger_yaml

with open('lineage.yml', 'w') as lineage: lineage.write(yaml_out)

# Deploy Challenger
project_details = cml.get_project({})
project_id = project_details["id"]

# Get Default Engine Details
default_engine_details = cml.get_default_engine({})
default_engine_image_id = default_engine_details["id"]

# Create Model
example_model_input = {
  "recency": "2",
  "history": "3",
  "used_discount": "0",
  "used_bogo": "1",
  "is_referral": "1",
  "channel_Phone": "1",
  "channel_Web": "1",
  "offer_Discount": "0",
  "offer_No Offer": "1"
}

create_model_params = {
    "projectId": project_id,
    "name": "Challenger " + run_time_suffix,
    "description": "A/B Test " + run_time_suffix,
    "visibility": "private",
    "enableAuth": False,
    "targetFilePath": "model_template.py",
    "targetFunctionName": "predict",
    "engineImageId": default_engine_image_id,
    "kernel": "python3",
    "examples": [
        {
            "request": example_model_input,
            "response": {}
        }],
    "cpuMillicores": 1000,
    "memoryMb": 2048,
    "nvidiaGPUs": 0,
    "replicationPolicy": {"type": "fixed", "numReplicas": 1},
    "environment": {}}

new_model_details = cml.create_model(create_model_params)
access_key = new_model_details["accessKey"]  # todo check for bad response
model_id = new_model_details["id"]

print("New model created with access key", access_key)

#Dump access key
import sqlite3
conn = sqlite3.connect('keys.db')
pd.DataFrame(new_model_details)[['id','accessKey']].drop_duplicates().reset_index(drop=True).to_sql(name='keys', con=conn, index=False, if_exists='append')
pd.read_sql('select * from keys', conn)

#spark.stop()

