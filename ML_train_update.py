from datawriteread import *
from ReceptorNeuralNetwork import *
from IdealDirection import *
from sklearn.preprocessing import MinMaxScaler

distance= [2,2.5,3,3.5,4,4.5,5,5.5,6]
direction_sphcoords = pick_direction(0,10) #same as sourcenum
#instances = np.arange(1,51,1)
X = read_datafile('Xseed_distance_big=2')
Xfinal = np.zeros((len(distance)*len(X),len(direction_sphcoords)))
Yfinal = np.zeros((len(distance)*len(X),len(direction_sphcoords)))

for i in distance:
    X = read_datafile('Xseed_distance_big='+str(i))
    Y = read_datafile('Yseed_distance_big='+str(i))
    indx= distance.index(i)
    Xfinal[indx*len(X):indx*len(X)+len(X),:]= X
    Yfinal[indx*len(Y):indx*len(Y)+len(Y),:]= Y
        
training_x, predict_x,training_y, predict_y = train_test_split(Xfinal, Yfinal)
mlp = train(training_x, training_y, layers_tuple = (19), max_iterations=20000, alph=0.0001,solve='adam')
acc, probs, score = test(mlp, predict_x, predict_y,direction_sphcoords,0.25)
print(acc)
save_neural_network(mlp, 'Total_mlp')
write_datafile('X_total_distances',params=[0],data=Xfinal)
write_datafile('Y_total_distances',params=[0],data=Yfinal)