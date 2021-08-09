from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
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

def accuracy(true_y, predicted_y):
    true_y = list(true_y)
    predicted_y = list(predicted_y)
    return sum(x == y for x, y in zip(true_y, predicted_y))/len(true_y)

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
    print("Accuracy of MLPClassifier : ", accuracy(predict_y, pred))
    print("Probabilities of each direction : ", direction_probabilities(mlp, predict_x))
    
def save_neural_network(mlp, distance=None,rate=None,diffusion=None,seed=None,cutoff=None,events=None,iterations=None):
    filename = 'MLPClassifier'
    if distance != None:
        filename += ' -d '+str(distance)
    if rate != None:
        filename += ' -r '+str(rate)
    if seed != None:
        filename += ' -S '+str(seed)
    if cutoff != None:
        filename += ' -c '+str(cutoff)
    if events != None:
        filename += ' -E '+str(events)
    if iterations != None:
        filename += ' -N '+str(iterations)
    if diffusion != None:
        filename += ' -s '+str(diffusion)
    pickle.dump(mlp, open(filename, 'wb'))
    return filename
    
    
def load_neural_network(filename):
    restored_mlp = pickle.load(open(filename, 'rb'))
    return restored_mlp
          
def direction_probabilities(mlp, X):
    return mlp.predict_proba(X)
           

if __name__ == '__main__':
     # new main for paula to use with dynamic filename, corrected accuracy function and proba of each class function

    X =
    Y = 
    training_x, training_y, predict_x, predict_y = separate_train_set(X, Y)
    mlp = train(training_x, training_y, layers_tuple = (6,4), max_iterations=1000)

    filename = save_neural_network(mlp, distance=10,rate=None,diffusion=None,seed=None,cutoff=None,events=None,iterations=None)
    restored_mlp = load_neural_network(filename)
    print(filename)
    
    test(restored_mlp, predict_x, predict_y)
