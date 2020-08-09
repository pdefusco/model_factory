import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

import pickle
from joblib import dump, load
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
grid = {'learning_rate': [0.1, 1]}

# Simplified Gridsearch... To do: unroll more gs metrics and track them
gs = GridSearchCV(clf, grid, scoring='accuracy')
gs.fit(X_train, y_train)

results = dict()
results['best_score'] = gs.best_score_
results['best_estimator'] = gs.best_estimator_
results['scorer'] = gs.best_score_
#results['best_parameters'] = gs.best_params_

results = pd.DataFrame(results, index=[0])

#Todo - do this more rigourously following best practices. Pull more metrics out. 
#Todo - could pass more scoring criteria from calling script, dynamically, even creating custom scoring functions
print("Best Accuracy Score")
print(results)

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

dump(clf, "models/clf_"+run_time_suffix+".joblib") 


# Tracking metric but if need to reuse the metric it's better to write to file?
cdsw.track_metric("Best Accuracy", results['best_score'])

# Tracking metric in db
import sqlite3
conn = sqlite3.connect('experiments.db')

results.to_sql(name='experiment_results', con=conn)
#pd.read_sql('select * from experiment_results', conn)