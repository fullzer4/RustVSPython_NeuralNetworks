import pandas as pd
import os

def getdata():
    
    home = os.getcwd()

    print(home)

    dataTest = pd.read_csv(f"{home}/data/test.csv").to_numpy()
    dataTrain = pd.read_csv(f"{home}/data/train.csv").to_numpy()
    
    return [dataTest, dataTrain]
