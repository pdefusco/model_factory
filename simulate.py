import requests
import pandas as pd
import numpy as np
import json
import random

def simulate_requests():

  #Creating fake instances
  instances = {} 

  instances['recency'] = random.randint(0,2)
  instances['history'] = random.randint(0,3)
  instances['used_discount'] = random.randint(0,1)
  instances['used_bogo'] = random.randint(0,1)
  instances['is_referral'] = random.randint(0,1)
  instances['channel_Phone'] = random.randint(0,3)
  instances['offer_Discount'] = random.randint(0,1)
  instances['offer_No Offer'] = random.randint(0,1)

  data = {}
  data['accessKey'] = 'm99j3w8hjto9mo29cqf8i5ir9b7betvu'
  data['request'] = instances

  json_object = json.dumps(data, indent = 4)

  #Submitting to AB_test.py
  r = requests.post('http://modelservice.ml-2f4cffbb-91e.demo-aws.ylcu-atmi.cloudera.site/model', 
                    data=json_object, headers={'Content-Type': 'application/json'})
  
  return {print(r.text)}

for i in range(1000):
  simulate_requests()