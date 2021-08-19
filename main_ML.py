from ReceptorNeuralNetwork import *
from datacreation import *
from IdealDirection import *
import numpy as np
#import pickle
"""
######## data ##################
#direction picking for cell: 
#create some kind of print command for user to select the directions
direction_sphcoords = pick_direction(0, 10)

#create data
X, Y = datacreate(direction_sphcoords, receptornum=10,particlenum=20)

print(X)
print(Y)
pickle.dump(X, open('X', 'wb'))
pickle.dump(Y, open('Y', 'wb')) 
"""

direction_sphcoords = pick_direction(0, 10)


#X = pickle.load(open('X', 'rb'))
#Y = pickle.load(open('Y', 'rb'))
#print(X.shape)
#print(Y.shape)
mysource_num = 20
myreceptornum = 10
myparticlenum = 25
myrecepsurface_ratio = 10
myradiusexact = 1
mydiffusionexact = 0.5
myrateexact = 0.1
myfirst_layer = 6
mysecond_layer = 4
myfrac_area = 0.4
myreceptor_seed = 1
myuse_seed = 1
mymax_iterations = 10000
print('# ML testing by Johannes on 16 Aug 2021')
print('# used parameters are')
#print('# direction_sphcoords '+str(direction_sphcoords))
print('# receptornum = '+str(myreceptornum))
print('# range of seeds for source creations: 1 to '+str(mysource_num))
print('# particlenum = is varying')
print('# recepsurface_ratio = '+str(myrecepsurface_ratio))
print('# radiusexact = '+str(myradiusexact))
print('# diffusionexact = '+str(mydiffusionexact))
print('# rateexact = '+str(myrateexact))
print('# used seed for receptors '+str(myreceptor_seed))
print('# used seed for everything else '+str(myuse_seed))
print('# size of first hidden layer = '+str(myfirst_layer))
print('# size of second hidden layer = '+str(mysecond_layer))
print('# fraction of sphere area considered under soft accuracy measure '+str(myfrac_area))
print('# max number of iterations for Neural Network to converge '+str(mymax_iterations))

print('# 1st column: particlenum, 2nd column: harsh accuracy, 3rd column: soft accuracy')
for myparticlenum in [10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50]:
    X1, Y1 = datacreate(direction_sphcoords, receptornum=myreceptornum, recepsurface_ratio=myrecepsurface_ratio,particlenum=myparticlenum,sourcenum=mysource_num, radiusexact=myradiusexact,diffusionexact=mydiffusionexact,rateexact=myrateexact,receptor_seed = myreceptor_seed,use_seed = myuse_seed)
    #pickle.dump(X1, open('X1', 'wb'))
    #pickle.dump(Y1, open('Y1', 'wb')) 
    #X1 = pickle.load(open('X1', 'rb'))
    #Y1 = pickle.load(open('Y1', 'rb'))
    #print(Y1)
    #print(X1)
    training_x, training_y, predict_x, predict_y = separate_train_set(X1, Y1)

    mlp = train(training_x, training_y, layers_tuple = (myfirst_layer,mysecond_layer), max_iterations=mymax_iterations)
    filename = save_neural_network(mlp)
    #restored_mlp = load_neural_network(filename)
    #print(filename)

    acc, directprob = test(mlp,direction_sphcoords,myfrac_area,myradiusexact, predict_x, predict_y)
    #print(X1)
    #print(Y1)
    #print('# soft accuracy = '+str(acc[0]))
    #print('# harsh accuracy = '+str(acc[1]))
    print(str(myparticlenum)+'\t'+str(acc[1])+'\t'+str(acc[0]))

