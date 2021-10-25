from ast import NodeVisitor
from os import read, write
from numpy.core.fromnumeric import _all_dispatcher
from numpy.core.function_base import linspace, logspace
from numpy.lib.function_base import diff
from ReceptorNeuralNetwork import *
from datacreate import *
from IdealDirection import *
from datawriteread import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

ranges = np.arange(10,401,10) #particles/receptors/distances/diffusion (c.f. datafiles.py)
instances = np.arange(1,21,1)
filenameX = 'Xseed_particlenum_update='
filenameY = 'Yseed_particlenum_update='

name_acctot = 'accuracy_total_particle19-0.1'
name_acc25 = 'accuracy0.25_total_particles19-0.1'
name_acc35 = 'accuracy0.35_total_particles19-0.1'
name_acc5 = 'accuracy0.5_total_particles19-0.1'

###################### particles/receptors/distances/diffusion ############################
direction_sphcoords = pick_direction(0,10) 

#...t is for the training set accuracy:
#accuracyt = np.zeros((len(instances),len(ranges)))

accuracytotal = np.zeros((len(instances),len(ranges)))
accuracynn25 = np.zeros((len(instances),len(ranges)))
accuracynn35 = np.zeros((len(instances),len(ranges)))
accuracynn50 = np.zeros((len(instances),len(ranges)))

for j in instances: 
    accur =[]
    #accurt=[]
    acn25 = []
    acn35 = []
    acn50 = []
   
    for i in ranges:
        X = read_datafile(filenameX+str(i))
        Y = read_datafile(filenameY+str(i))
        training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
        mlp = train(training_x, training_y, layers_tuple =(19), max_iterations=50000, alph=0.1)
        #^selected parameters after gridsearch 

        pred = predict(mlp, predict_x)
        acc = accuracy_score(predict_y,pred)
        #predt = predict(mlp, training_x)
        #acct = accuracy_score(training_y,predt) #training accuracy
        accnn25 = nearest_neighbours_accuracy(direction_sphcoords,predict_y,pred,0.25)
        accnn35 = nearest_neighbours_accuracy(direction_sphcoords,predict_y,pred,0.35)
        accnn50 = nearest_neighbours_accuracy(direction_sphcoords,predict_y,pred,0.50)
        accur.append(acc)
        #accurt.append(acct) #training accuracy
        acn25.append(accnn25)
        acn35.append(accnn35)
        acn50.append(accnn50)
    accuracytotal[j-1,:]=accur
    #accuracyt[j-1,:]=accurt  #training accuracy
    accuracynn25[j-1,:]=acn25
    accuracynn35[j-1,:]=acn35
    accuracynn50[j-1,:]=acn50
write_datafile(name_acctot,params=[0],data=accur)
write_datafile(name_acc25,params=[0],data=accuracynn25)
write_datafile(name_acc35,params=[0],data=accuracynn35)
write_datafile(name_acc5,params=[0],data=accuracynn50)

############ alphas test for particlenum 400 (figures in thesis report page 30) #######################

alpha = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1]
accuracytotal = np.zeros((len(instances),len(alpha)))

X = read_datafile('Xseed_particlenum_update='+str(400))
Y = read_datafile('Yseed_particlenum_update='+str(400))
for j in instances: 
    accur =[]
    accurt=[]
    for i in alpha:
        
        training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
        mlp = train(training_x, training_y, layers_tuple =(19), max_iterations=50000, alph=i)
        pred = predict(mlp, predict_x)
        acc = accuracy_score(predict_y,pred)
        accur.append(acc)
    accuracytotal[j-1,:]=accur
 
write_datafile('accuracyalphas400',params=[0],data=accuracytotal)

######### NN hidden layer size test ###################################
readX = 'Xseed_distance_big_update=2'
readY = 'Yseed_distance_big_update=2'
nodes = np.arange(3,30)
instances = np.arange(1,51,1)
accuracytotal = np.zeros((len(instances),len(nodes)))

for j in instances:
    accur =[]
    for i in nodes:
        X = read_datafile(readX) 
        Y = read_datafile(readY)
        training_x, predict_x,training_y, predict_y = train_test_split(X, Y)
        mlp = train(training_x, training_y, layers_tuple =(i), max_iterations=50000, alph=0.1, solve ='lbfgs')
        acc, probs, score = test(mlp, predict_x, predict_y,direction_sphcoords,0.25)
        accur.append(acc)
    accuracytotal[j-1,:]=accur
write_datafile('accuracy_nodes',params=[0],data=accuracytotal)

#for big size the alpha is 0.1 and the conclusion is that 19 is better

