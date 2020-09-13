import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from joblib import dump, load

import pickle
import datetime
import cdsw

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://demo-aws-1/")\
    .config("spark.hadoop.yarn.resourcemanager.principal",os.getenv("HADOOP_USER_NAME"))\
    .config("spark.executor.instances", 2)\
    .config("spark.executor.cores", 2)\
    .getOrCreate()
    
#To do: take in new data rather than old table
df = spark.sql("select * from default.historical_customer_interactions")

df = df.toPandas()

df = pd.get_dummies(df, columns=["channel", "offer"], drop_first=True)
df = df.drop(columns=['zip_code'])

y = df['conversion']
X = df.drop(columns=['conversion'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Using experiment instance set in CICD.py
data_dir = "data"
clf = GradientBoostingClassifier()
grid = {'learning_rate': [0.1, 1], 'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 10]}
cv = int(3)

# Simplified Gridsearch... To do: unroll more gs metrics and track them
gs = GridSearchCV(clf, grid, cv=cv, scoring='accuracy', n_jobs=-1)
gs.fit(X_train, y_train)

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

results = dict()
results['timestamp'] = run_time_suffix
results['clf'] = clf.__class__.__name__
results['best_score'] = gs.best_score_
#results['best_params'] = [gs.best_params_]
results['n_splits'] = gs.n_splits_
results['scorer'] = gs.scorer_
#results['cv_results'] = [gs.cv_results_]
#results['grid'] = [grid]

results_df = pd.DataFrame(results, index=[0])

results_df['timestamp'] = results_df['timestamp'].astype(str)
results_df['clf'] = results_df['clf'].astype(str)
results_df['best_score'] = results_df['best_score'].astype(float)
#results_df['best_params'] = results_df['best_params'].astype(str)
results_df['n_splits'] = results_df['n_splits'].astype(int)
results_df['scorer'] = results_df['scorer'].astype(str)
#results_df['cv_results'] = results_df['cv_results'].astype(str)
#results_df['grid'] = results_df['grid'].astype(str)

print("Best Accuracy Score")
print(results)
cdsw.track_metric("Best Accuracy Score", gs.best_score_)

spark.sql("CREATE TABLE IF NOT EXISTS default.experiment_outcomes (TIMESTAMP STRING, CLASSIFIER STRING, \
            BEST_SCORE FLOAT, N_SPLITS INT, SCORER STRING)")

#spark.sql("CREATE TABLE IF NOT EXISTS default.experiment_outcomes (TIMESTAMP STRING, CLASSIFIER STRING, \
#            BEST_SCORE INTEGER, BEST_PARAMS STRING, N_SPLITS INT, SCORER STRING, CV_RESULTS STRING, GRID STRING)")

experiments_df = spark.createDataFrame(results_df)
  
experiments_df.write.insertInto("default.experiment_outcomes",overwrite = False) 