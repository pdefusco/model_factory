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
data_dir = str(data)
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
results['clf'] = clf
results['best_estimator'] = gs.best_estimator_
results['best_score'] = gs.best_score_
results['best_params'] = [gs.best_params_]
results['n_splits'] = gs.n_splits_
results['scorer'] = gs.scorer_
results['cv_results'] = [gs.cv_results_]
results['grid'] = [grid]

results_df = pd.DataFrame(results, index=[0])

dump(clf, "models/clf_"+run_time_suffix+".joblib") 

# Tracking metric but if need to reuse the metric it's better to write to file?
cdsw.track_metric("Best Accuracy", results['best_score'])


from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://demo-aws-1/")\
    .config("spark.hadoop.yarn.resourcemanager.principal",os.getenv("HADOOP_USER_NAME"))\
    .getOrCreate()
    
spark.sql("CREATE TABLE IF NOT EXISTS default.experiment_outcomes (BEST_SCORE FLOAT, N_SPLITS INT)")

experiments_df = spark.createDataFrame(results_df)
    
experiments_df.write.insertInto("default.experiment_outcomes",overwrite = False)    
    

