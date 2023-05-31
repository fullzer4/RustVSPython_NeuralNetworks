from data.data import getdata
from func.func import init_params
import numpy as np

def main():

    data = getdata()

    data = np.array(data)

    m, n = data.shape
    np.random.shuffle(data)

    dataTest = data[0:1000].T
    x_test = dataTest[1:n]
    y_test = dataTest[0]

    dataTrain = data[1000:m].T
    x_train = dataTrain[1:n]
    y_train = dataTrain[0]
    
    W1, b1, W2, b2 = init_params()

main()
