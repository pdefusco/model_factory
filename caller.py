from Experiment import Experiment
from sklearn.linear_model import LogisticRegression

experiments_dict = dict()
experiments_dict['LR'] = {'data_dir':'data', 'clf':LogisticRegression(), 'grid':{'C':[.1,1]}}
experiment = Experiment(experiments_dict['LR']['data_dir'], experiments_dict['LR']['clf'], experiments_dict['LR']['grid'])

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
    "script": "new_experiment.py",
    "arguments": " ",
    "kernel": "python3",
    "cpu": 1,
    "memory": 2,
    "project": 2,
    "name":"new experiment name"
}



x = 5
import called

