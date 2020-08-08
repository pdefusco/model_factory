import sklearn
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import os


class Experiment():
    
    def __init__(self, datadir, classifier, grid):
        self.datadir = datadir
        self.classifier = classifier
        self.grid = grid 
        
    def show_experiment(self):
        print(self.classifier.__class__.__name__)
        
        