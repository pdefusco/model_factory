import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import pickle
from joblib import dump, load
import datetime

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://demo-aws-1/")\
    .config("spark.hadoop.yarn.resourcemanager.principal",os.getenv("HADOOP_USER_NAME"))\
    .config("spark.executor.instances", 2)\
    .config("spark.executor.cores", 2)\
    .getOrCreate()
    
df = spark.read.option('inferschema','true').csv(
  "s3a://demo-aws-1/datalake/model_factory_demo",
  header=True,
  sep=',',
  nullValue='NA'
)

df.write.format('parquet').mode("overwrite").saveAsTable('default.historical_customer_interactions')

df = df.toPandas()

df = pd.get_dummies(df, columns=['channel', 'offer'], drop_first=True)
df = df.drop(columns=['zip_code'])

y = df['conversion']
X = df.drop(columns=['conversion'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = GradientBoostingClassifier(random_state=0)

clf.fit(X_train, y_train)

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

dump(clf, 'models/clf_'+run_time_suffix+'.joblib') 