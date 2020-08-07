import pandas as pd

pd.read_csv('data/new_interactions_2020-08-06 23:10:48.086433.csv')

type(pd.json_normalize({"recency":"2", "history":"3", "used_discount":"0", "used_bogo":"1", "is_referral":"1", "channel":"1", "offer":"0"}))


from collections import ChainMap

data = pd.DataFrame({"recency":"2", "history":"3", "used_discount":"0", "used_bogo":"1", "is_referral":"1", "channel_Phone":"1", "channel_Web":"1", "offer_Discount":"0", "offer_No Offer":"1"}

, index=[0])

clf = load('models/clf.joblib') 



type(int(clf.predict(data)[0]))

clf.predict_proba(data)[0][0]

data.astype(float).dtypes

type(data.to_json())

pd.DataFrame(data)