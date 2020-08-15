
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://demo-aws-1/")\
    .config("spark.hadoop.yarn.resourcemanager.principal",os.getenv("HADOOP_USER_NAME"))\
    .getOrCreate()
    
    
df = spark.sql("SELECT * FROM default.experiment_outcomes")

experiments_df = df.select("*").toPandas()

print(experiments_df)

experiments_df.to_csv("experiments_summary/experiments_summary.csv", index=False)

spark.stop()