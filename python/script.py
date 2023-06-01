from data.data import getdata
from func.func import grad_descent, fof_propagation, get_predictions
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
    
    W1, b1, W2, b2 = grad_descent(x_train, y_train, 100, 0.1)

    def make_predictions(X, W1, b1, W2, b2):
        _, _, _, A2 = fof_propagation(W1, b1, W2, b2, X)
        predictions = get_predictions(A2)
        return predictions

    def test_prediction(index, W1, b1, W2, b2):
        current_image = x_train[:, index, None]
        prediction = make_predictions(x_train[:, index, None], W1, b1, W2, b2)
        label = y_train[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

    test_prediction(0, W1, b1, W2, b2)
    test_prediction(1, W1, b1, W2, b2)
    test_prediction(2, W1, b1, W2, b2)
    test_prediction(3, W1, b1, W2, b2)
main()
