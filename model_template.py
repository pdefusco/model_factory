import sklearn
import pandas as pd
import numpy as np
from joblib import load
import cdsw
import os

clf = load('models/challengers/clf.joblib') 

##### Model Tracking Decorator - use to track model with PostgresDB
@cdsw.model_metrics
def predict(args):
  
    data = pd.DataFrame(args, index=[0])
    data = data.astype(float)
    prediction = clf.predict(data)
    probability = clf.predict_proba(data)
    
    # Track inputs
    #cdsw.track_metric('input_data', data)
    
    # Track our prediction
    cdsw.track_metric('probability', int(probability[0][0]))
    
    # Track explanation
    cdsw.track_metric('explanation', int(prediction[0]))
    
    return {
        'prediction': prediction,
        'probability': probability
        }

##### Example model input: 
##### {"recency":"2", "history":"3", "used_discount":"0", "used_bogo":"1", "is_referral":"1", "channel_Phone":"1", "channel_Web":"1", "offer_Discount":"0", "offer_No Offer":"1"}

