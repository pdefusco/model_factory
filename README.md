# Model Factory

### Objective: create a CI/CD pipeline using ML Ops for Next Best Action

##### Components: CML


1. Model Development.ipynb
      - This notebook representa an initial research effort by the data scientist
      - The original model is then deployed manually via CML Models

2. CICD.py launches a set of parlallel experiments programmatically. 
      - The file contains a predetermined set of classifiers along with respective hyperparameters
      - Each classifier and related hyperparameters are written to the experiment_template.py iteratively
      - experiment_template.py is launched as an individual experiment, then the new classifier is overwritten to the file and a new experiment is launched again
      - Experiment outcomes are stored in a Hive table
      - Experiment classifeirs are tracked via the cdsw.track_file() method. 
   
3. experiments_dashboard.py is a CML Visual Application used to analyze all experiment runs.
      - read_experiment_outcomes.py is executed every 30 minutes as a recurrent job. 
      - It reads from the Hive table where all experiments are stored, and overwrites them to the experiments_summary.csv file in experiments_summary
      - The dashboard is therefore refreshed on an ongoing basis to reflect the most recent experiments.
      - The data scientist can pick which experiments have yielded the most interesting outcomes and investigate hyperparameters, CV scores, etc.

4. The best experiment is deployed as a ML model - WIP
      - The continuous_deployment.py script reads the best classifier from the experiments table and retrains it on the entire training set.
      - The newly ready classifier is deployed programmatically via a job which is scheduled to run 2 hours after the experiments are complete.
      - The DevOps team could optionally use the experiments landing page to decide which experiment classifiers should be promoted to the project files and deployed.
      
5. The new model is monitored by CML with a new Visual Application - WIP
      - The CML track_aggregate_metrics method is used for model inference
      
      
##### Notice:

As mentioned in the last bullet point in step 4, this step could either be done entirely or partially manually.  


##### Future Components: CML, Kudu, NiFi

Future enhancements:

- Set experiment resources (CPU/memory) dynamically in order to allocate more to more computationally complex algorithms
- Use NiFi to generate new customer interaction data
- Use Kudu to store new customer interaction data
- Use CML track_delayed_metrics to capture and updated model ground truth from delayed customer responses