import pandas as pd
import os
import numpy as np

def getdata():
    
    home = os.getcwd()

    print(home)

    dataTest = pd.read_csv(f"{home}/data/test.csv")
    dataTrain = pd.read_csv(f"{home}/data/train.csv")
    
    dataTest = np.array(dataTest)
    dataTrain = np.array(dataTrain)

    return [dataTest, dataTrain]
