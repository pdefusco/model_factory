#### SIMULATE NEW DATA ####

# This script will create new data with the Faker library
# This is designed to provide new data for the CI/CD pipline 
# The pipeline will automatically retrain the model and promote it dynamically

import pandas as pd
import numpy as np
import os
import random, string, decimal
from datetime import datetime


# Loading the original data to get data heuristics so we can simulate the data realistically
df = pd.read_csv('data/conversion.csv')

rec_max = df['recency'].max()
rec_min = df['recency'].min()
history_max = df['history'].max()
history_min = df['history'].min()
used_discount_max = df['used_discount'].max()
used_discount_min = df['used_discount'].min()
used_bogo_max = df['used_bogo'].max()
used_bogo_min = df['used_bogo'].min()
is_referral_max = df['is_referral'].max()
is_referral_min = df['is_referral'].min()
conversion_max = df['conversion'].max()
conversion_min = df['conversion'].min()

# Attributes we are going to create
# recency	history	used_discount	used_bogo	zip_code	is_referral	channel	offer	conversion
# zip_code not needed bcause we dropped it in the Model Development notebook

fake_data = {}
fake_data_length = 500

fake_data['recency'] = [random.randint(rec_min,rec_max) for i in range(fake_data_length)]
fake_data['history'] = [float(decimal.Decimal(random.randrange(round(used_discount_min), round(history_max)))/100) for i in range(fake_data_length)]
fake_data['used_discount'] = [random.randint(used_discount_min,used_discount_max) for i in range(fake_data_length)]
fake_data['used_bogo'] = [random.randint(used_bogo_min,used_bogo_max) for i in range(fake_data_length)]
fake_data['is_referral'] = [random.randint(is_referral_min,is_referral_max) for i in range(fake_data_length)]
fake_data['channel'] = [random.choice(list(df['channel'].unique())) for i in range(fake_data_length)]
fake_data['offer'] = [random.choice(list(df['offer'].unique())) for i in range(fake_data_length)]
fake_data['conversion'] = [random.randint(conversion_min,conversion_max) for i in range(fake_data_length)]

fake_df = pd.DataFrame(fake_data)

fake_df.head()

#Filename

filename = 'new_interactions_{}'.format(datetime.now())

fake_df.to_csv('data/'+filename+'.csv', index=False)