import pandas as pd
import os

def getdata():
    
    dataTest = pd.read_csv("../../mnist/test.csv")
    dataTrain = pd.read_csv("../../mnist/train.csv")
    
    return [dataTest, dataTrain]

