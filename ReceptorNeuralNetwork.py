from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def fit(X, y):
    X = MinMaxScaler().fit_transform(X)
    mlp = MLPClassifier(random_state=0, max_iter=400, solver='sgd', learning_rate='constant',
                        momentum=0, learning_rate_init=0.2)

    mlp.fit(X, y)
    return mlp

def predict(mlp: MLPClassifier, X):
    X = MinMaxScaler().fit_transform(X)
    y = mlp.predict(X)
    return y

def nearest_receptor(source, receptors):
    #input x, y, z of source and receptors
    separation_vector = receptors - source
    distance = (np.linalg.norm(row) for row in separation_vector)
    return np.where(distance == np.amin(distance))
    
    
    
    
    
