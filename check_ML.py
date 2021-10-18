from ReceptorNeuralNetwork import *
from datawriteread import *
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html

#better to check and do average between many

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
"""
parameters ={'solver': ['lbfgs'],'alpha':[0.0001,0.001,0.01,0.1,0.5,1.0,1.5,2.0,2.5,3.0], 'hidden_layer_sizes' : np.arange(10,20),'early_stopping':[True], \
    'max_iter': [20000]}
#parameters = {'solver': ['adam'], 'max_iter': [1000,2000,3000,4000,5000 ], \
    #'alpha':np.logspace(-1, 1, 5), \
    #'hidden_layer_sizes':np.arange(15, 25), 'beta_1':np.linspace(0.1, 0.9, 5), \
    #'beta_2' : np.linspace(0.1, 0.999, 5), 'epsilon': np.logspace(-8,-2,5),  'early_stopping':[True]}
clf = GridSearchCV(MLPClassifier(), parameters)
X = read_datafile('Xseed_distance=2') #distanceonly when using 'adam' (bottom results)
Y = read_datafile('Yseed_distance=2')
X = MinMaxScaler().fit_transform(X)
clf.fit(X, Y)
report(clf.cv_results_)

Model with rank: 1
Mean validation score: 0.755 (std: 0.032)
Parameters: {'alpha': 0.001, 'early_stopping': True, 'hidden_layer_sizes': 16, 'max_iter': 10000, 'solver': 'lbfgs'}

Model with rank: 2
Mean validation score: 0.752 (std: 0.031)
Parameters: {'alpha': 0.1, 'early_stopping': True, 'hidden_layer_sizes': 14, 'max_iter': 10000, 'solver': 'lbfgs'}

Model with rank: 2
Mean validation score: 0.752 (std: 0.036)
Parameters: {'alpha': 0.1, 'early_stopping': True, 'hidden_layer_sizes': 17, 'max_iter': 10000, 'solver': 'lbfgs'}
"""
parameters ={'solver': ['adam'],'alpha':[0.0001,0.001,0.01,0.1,0.5,1.0,1.5,2.0,2.5,3.0], 'hidden_layer_sizes' : np.arange(10,20),'early_stopping':[True], \
    'max_iter': [20000]}
#parameters = {'solver': ['adam'], 'max_iter': [1000,2000,3000,4000,5000 ], \
    #'alpha':np.logspace(-1, 1, 5), \
    #'hidden_layer_sizes':np.arange(15, 25), 'beta_1':np.linspace(0.1, 0.9, 5), \
    #'beta_2' : np.linspace(0.1, 0.999, 5), 'epsilon': np.logspace(-8,-2,5),  'early_stopping':[True]}
clf = GridSearchCV(MLPClassifier(), parameters)
X = read_datafile('Xseed_distanceonly=2') #distanceonly when using 'adam' (bottom results)
Y = read_datafile('Yseed_distanceonly=2')
X = MinMaxScaler().fit_transform(X)
clf.fit(X, Y)
report(clf.cv_results_)
"""
Model with rank: 1
Mean validation score: 0.604 (std: 0.249)
Parameters: {'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': 19, 'max_iter': 20000, 'solver': 'adam'}

Model with rank: 2
Mean validation score: 0.374 (std: 0.298)
Parameters: {'alpha': 0.0001, 'early_stopping': True, 'hidden_layer_sizes': 16, 'max_iter': 20000, 'solver': 'adam'}

Model with rank: 3
Mean validation score: 0.364 (std: 0.293)
Parameters: {'alpha': 0.001, 'early_stopping': True, 'hidden_layer_sizes': 13, 'max_iter': 20000, 'solver': 'adam'}
"""

