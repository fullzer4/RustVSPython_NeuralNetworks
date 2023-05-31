from data.data import getdata
import numpy as np
import pandas as pd 

def main():

    data = getdata()
    # [0] test dataset
    # [1] train dataset

    dataTest = np.array(data[0])
    dataTrain = np.array(data[1])

    m, n = dataTrain.shape
    np.random.shuffle(dataTrain)

    

main()
