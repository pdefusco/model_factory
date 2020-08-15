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
from sklearn.ensemble import RandomForestClassifier

HOST = os.getenv("CDSW_API_URL").split(":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split("/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT") 

# Instantiate API Wrapper
# Passing API key directly is better
cml = CMLBootstrap(HOST, USERNAME, "uuc48l0gm0r3n2mib27voxazoos65em0", PROJECT_NAME)

# Get Project Details
project_details = cml.get_project({})
project_id = project_details["id"]

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

###### Experiments ######

# Create Models for Experiments

experiments_dict = {}

#Add more estimators at will!
experiments_dict['LR'] = {'data_dir':'data', 'clf':LogisticRegression(), 'grid':{'C':[.1,1], 'solver':['lbfgs', 'liblinear'], 'class_weight':['None','balanced']}, 'cv':'3'}
experiments_dict['GBC'] = {'data_dir':'data', 'clf':GradientBoostingClassifier(), 'grid':{'learning_rate':[.1,1], 'n_estimators':[50,100,150], 'max_depth':[3,5,10]}, 'cv':'3'}
experiments_dict['RFC'] = {'data_dir':'data', 'clf':RandomForestClassifier(),'grid':{'min_impurity_decrease':[0.2, 0.4], 'n_estimators':[50,100,150], 'max_depth':[3,5,10], 'n_jobs':[-1]}, 'cv':'3'}

# Run Experiments
def create_experiment_script(data_dir, clf, grid, cv):

  with open("experiment_template.py", "r") as rfile:
    data = rfile.read()

  data = data.format(str(data_dir), str(clf), str(grid), str(cv))

  run_time_suffix = datetime.datetime.now()
  run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

  write_file = "experiments/new_experiment_"+run_time_suffix+".py"
  with open(write_file, "w") as wfile:
    wfile.write(data)
    
  return write_file

def run_all_experiments(experiments_dict, cpu, memory):
#To Do clean up
  for k, v in experiments_dict.items():
    experiment_name = create_experiment_script(v["data_dir"], v["clf"], v["grid"], v["cv"])

    run_experiment_params = {
      "size": {
          "id": 1,
          "description": "{} vCPU / {} GiB Memory".format(cpu, memory),
          "cpu": cpu,
          "memory": memory,
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
      "cpu": cpu,
      "memory": memory,
      "project": str(project_id)
  }


    new_experiment_details = cml.run_experiment(run_experiment_params)

#Running all experiments in bulk
#TODO - assign experiment resources dynamically

run_all_experiments(experiments_dict, 2, 4)


