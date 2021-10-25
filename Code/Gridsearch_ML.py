from ReceptorNeuralNetwork import *
from datawriteread import *
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html

#Test MLPClassifier parameters that are best for the data X and Y
#Modify the data that is going to be checked or the parameters 

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

parameters ={'max_iter': [50000],'solver': ['lbfgs'],'alpha':[0.0001,0.001,0.01,0.1,0.5,1.0,1.5,2.0,2.5,3.0], 'hidden_layer_sizes' : np.arange(10,20),'early_stopping':[True]}
#parameters to be checked on a grid

clf = GridSearchCV(MLPClassifier(), parameters)
X = read_datafile('Xseed_distance_update=10') #data X
Y = read_datafile('Yseed_particle_update=10') #data Y
X = MinMaxScaler().fit_transform(X)
clf.fit(X, Y)
report(clf.cv_results_)


