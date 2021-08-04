from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import datasets
import pickle

def fit_mlp(X, Y, layers_tuple, max_iterations):
    #function to scale X to between 0 and 1 which is best for neural network, then creates mlp object 
    X = MinMaxScaler().fit_transform(X)
    mlp = MLPClassifier(hidden_layer_sizes=layers_tuple,random_state=0, max_iter=max_iterations, solver='sgd', learning_rate='constant',
                        momentum=0, learning_rate_init=0.2) 
    #layers_tuple: Each element in the tuple is the number of nodes at the ith position. 
    #Length of tuple denotes the total number of layers.
    mlp.fit(X, Y)
    
    return mlp

def predict(mlp: MLPClassifier, X):
    #function to scale X to between 0 and 1 and then use the Neural Network to produce predictions for Y
    X = MinMaxScaler().fit_transform(X)
    y = mlp.predict(X)
    
    return y

def accuracy(confusion_matrix):
   #function to calculate the accuracy of our NN
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   
   return diagonal_sum / sum_of_all_elements

def separate_train_set(X,Y):
    #if data is ordered use this to randomise it so every source position is trained on, then use first half of data to train the network
    #use second half of the data to predict if network is working
    num_rows, num_cols = np.shape(X)
    indices = np.random.choice(range(num_rows), num_rows, replace=False) 
    n_train = int(len(indices)/2)
    training_x = np.array(X)[indices[:n_train].astype(int)]
    training_y = np.array(Y)[indices[:n_train].astype(int)]
    predict_x = np.array(X)[indices[n_train:].astype(int)] 
    predict_y = np.array(Y)[indices[n_train:].astype(int)]
    
    return training_x, training_y, predict_x, predict_y

def train(training_x, training_y, layers_tuple, max_iterations):
    #function to train our neural network with half the data given
    mlp = fit_mlp(training_x, training_y, layers_tuple, max_iterations)
    return mlp

def test(mlp, predict_x, predict_y):
    pred = predict(mlp, predict_x)
    cm = confusion_matrix(pred, predict_y)
    print("Accuracy of MLPClassifier : ", accuracy(cm))
    
def save_neural_network(mlp, filename):
    pickle.dump(mlp, open(filename, 'wb'))
    
    
def load_neural_network(filename):
    restored_mlp = pickle.load(open(filename, 'rb'))
    return restored_mlp
          
         

if __name__ == '__main__':
    
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    training_x, training_y, predict_x, predict_y = separate_train_set(X, Y)
    mlp = train(training_x, training_y, layers_tuple = (6,4), max_iterations=1000)

    save_neural_network(mlp)
    restored_mlp = load_neural_network()
    test(restored_mlp, predict_x, predict_y)
