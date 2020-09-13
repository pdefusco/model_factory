## Execute A/B test between champion and challenger deployed models
## This script is deployed as a third model.
## It routes requests between the two models and executes the A/B test

import pandas as pd
import numpy as np
import requests
import random

from cmlbootstrap import CMLBootstrap
import datetime
import os, time

#Retrieve project info with CML library
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = "uuc48l0gm0r3n2mib27voxazoos65em0"
PROJECT_NAME = os.getenv("CDSW_PROJECT") 

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

#Retrieve model access keys for models that are compared

#Champion can be either hardcoded or can be the most recent model from the day it was deployed
#mvav17o0lwb9oogg3jlh8g7wqaw99e6w
champion_ak = "mvav17o0lwb9oogg3jlh8g7wqaw99e6w"

#Challenger is the most recent model deployed today
project_id = cml.get_project({})['id'] #get project ID
deployed_models_df = pd.DataFrame(cml.get_models({}))
challenger_ak = deployed_models_df[deployed_models_df['projectId'] == project_id]\
    .sort_values("createdAt", ascending=False)['accessKey'].iloc[0]

def route_request(args):
    
    coin = random.randint(0,1)
    
    #TODO: make this more coincise. Something messing up json
    #TODO: add more metadata about model chosen
    if coin == 0:
      r = requests.post('http://modelservice.ml-2f4cffbb-91e.demo-aws.ylcu-atmi.cloudera.site/model', data='{"accessKey":"mvav17o0lwb9oogg3jlh8g7wqaw99e6w","request":{"recency":"2","history":"3","used_discount":"0","used_bogo":"1","is_referral":"1","channel_Phone":"1","channel_Web":"1","offer_Discount":"0","offer_No Offer":"1"}}', headers={'Content-Type': 'application/json'})
    else:
      r = requests.post('http://modelservice.ml-2f4cffbb-91e.demo-aws.ylcu-atmi.cloudera.site/model', data='{"accessKey":"mdqv2oy7q98um7cpziqjundh48zx6gfu","request":{"recency":"2","history":"3","used_discount":"0","used_bogo":"1","is_referral":"1","channel_Phone":"1","channel_Web":"1","offer_Discount":"0","offer_No Offer":"1"}}', headers={'Content-Type': 'application/json'})
 
    # extracting response text 
    pastebin_url = r.text 
    
    return {
          print("The pastebin URL is:%s"%pastebin_url), 
          print(coin)
        }
  

