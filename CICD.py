####### Simplified CI/CD Pipeline #########

#    - Retrain the model with new interactions via automated experiments
#    - Evaluate model and promote it if performance is above threshold?
#    - Save model to model repo (model dir)

from cmlbootstrap import CMLBootstrap
import datetime
import os, time
from Experiment import Experiment

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

HOST = os.getenv("CDSW_API_URL").split(":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split("/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT") 

# Get Project Details
project_details = cml.get_project({})
project_id = project_details["id"]

# Instantiate API Wrapper
# Passing API key directly is better
cml = CMLBootstrap(HOST, USERNAME, "uuc48l0gm0r3n2mib27voxazoos65em0", PROJECT_NAME)

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

###### Experiments ######

# Create Models for Experiments

experiments_dict = {}

experiments_dict['LR'] = {'data_dir':'data', 'clf':LogisticRegression(), 'grid':{'C':[.1,1]}}
experiments_dict['GBC'] = {'data_dir':'data', 'clf':GradientBoostingClassifier(), 'grid':{'learning_rate':[.1,1]}}

# Run Experiments


def create_experiment_script(data_dir, clf, grid):

  with open("experiment_template.py", "r") as rfile:
    data = rfile.read()

  data = data.format(str(data_dir), str(clf), str(grid))

  run_time_suffix = datetime.datetime.now()
  run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

  write_file = "experiments/new_experiment_"+run_time_suffix+".py"
  with open(write_file, "w") as wfile:
    wfile.write(data)
    
  return write_file

#To Do - put this into a function, clean up

for k, v in experiments_dict.items():
  experiment_name = create_experiment_script(v["data_dir"], v["clf"], v["grid"])
  
  run_experiment_params = {
    "size": {
        "id": 1,
        "description": "1 vCPU / 2 GiB Memory",
        "cpu": 1,
        "memory": 2,
        "route": "engine-profiles",
        "reqParams": None,
        "parentResource": {
            "route": "site",
            "parentResource": None
        },
        "restangularCollection": True
    },
    "script": experiment_name,
    "arguments": " ",
    "kernel": "python3",
    "cpu": 1,
    "memory": 2,
    "project": str(project_id)
}
  
  
  new_experiment_details = cml.run_experiment(run_experiment_params)


