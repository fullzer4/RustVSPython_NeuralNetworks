import pandas as pd
import os
import numpy as np

def getdata():
    
    home = os.getcwd()
    
    data = pd.read_csv(f"{home}/data/train.csv")

    data = np.array(data)

    return data
