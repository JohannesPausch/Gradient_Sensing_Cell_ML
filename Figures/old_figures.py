import matplotlib as plt
import numpy as np
from numpy.lib.function_base import diff
from scipy.sparse import data
from vpython.vpython import color
from IdealDirection import *
from datawriteread import *
from ReceptorNeuralNetwork import *
"""
plt.rcParams.update({'font.size': 33})

#particlenum= [5,10,20,30,40,50,60,70,80,90,100]
particlenum= np.arange(10,401,10)
accuracy = read_datafile('update_accuracy_total_particle19-0.1')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
acc25 = read_datafile('accuracynn0.25_total_particles19-0.1')
del acc25[0]
accmean25 = np.mean(acc25, axis =0)
err25= np.std(acc25, axis=0)
acc35 = read_datafile('accuracynn0.35_total_particles19-0.1')
del acc35[0]
accmean35 = np.mean(acc35, axis =0)
err35= np.std(accmean35, axis=0)
acc5 = read_datafile('accuracynn0.50_total_particles19-0.1')
del acc5[0]
accmean5 = np.mean(acc5, axis =0)
err5= np.std(accmean5, axis=0)
plt.figure(figsize=(22,14))
print(np.where(accmean==np.max(accmean)))
#plt.errorbar(particlenum, accmean,yerr = err, color = 'black', marker = 'x', linestyle=' ', label = 'Harsh Accuracy')
#plt.errorbar(particlenum, accmean25,yerr=err25,marker='o', markersize=5,c = 'tomato', mfc = 'red',linestyle='-', label='NN Cap Size = 0.25')
#plt.errorbar(particlenum, accmean35,yerr=err35,markersize=5, c= 'lightgreen', mfc = 'green', linestyle='-',label='NN Cap Size = 0.35')
#plt.errorbar(particlenum, accmean5,yerr = err5,markersize=5, c = 'skyblue', mfc = 'blue', linestyle='-', label='NN Cap Size = 0.50')
plt.plot(particlenum, accmean,color = 'purple', marker = 'o', markersize=13,linestyle='-', linewidth=2, label= 'Harsh accuracy')
plt.fill_between(particlenum, accmean-err, accmean+err, edgecolor = 'purple', facecolor='#D2B4DE')
plt.plot(particlenum, accmean25,marker='o', markersize=13,c = 'red', mfc = 'red',linestyle='-',linewidth=2, label='n.n.a. cap ratio = 0.25')
plt.fill_between(particlenum, accmean25-err25, accmean25+err25,edgecolor = 'red', facecolor='#F5B7B1')
plt.plot(particlenum, accmean35,marker='o',markersize=13, c= 'green', mfc = 'green', linestyle='-',linewidth=2,label='n.n.a. cap ratio = 0.35')
plt.fill_between(particlenum, accmean35-err35, accmean35+err35,edgecolor = 'green', facecolor='#A9DFBF')
plt.plot(particlenum, accmean5,marker='o',markersize=13, c = 'blue', linestyle='-',linewidth=2, label='n.n.a. cap ratio = 0.50')
plt.fill_between(particlenum, accmean5-err5, accmean35+err5,edgecolor = 'blue', facecolor='#A9CCE3')
plt.legend()
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of particle events')
plt.savefig('particles19-0.1_NNerror2.png')
"""

plt.rcParams.update({'font.size': 35})

recept = read_datafile('recept')
del recept[0]

plt.figure(figsize=(22,14))
accuracy = read_datafile('update_accuracy_total_recept+10-0.act')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
acc25 = read_datafile('25update_accuracy_total_recept+10-0.1act')
del acc25[0]
accmean25 = np.mean(acc25, axis =0)
err25= np.std(acc25, axis=0)
acc35 = read_datafile('35update_accuracy_total_recept+10-0.1act')
del acc35[0]
accmean35 = np.mean(acc35, axis =0)
err35= np.std(acc35, axis=0)
acc5 = read_datafile('5update_accuracy_total_recept+10-0.1act')
del acc5[0]
accmean5 = np.mean(acc5, axis =0)
err5= np.std(acc5, axis=0)

del recept[0][np.array(np.where(np.array(recept)==20)[1])[0]]
del recept[0][np.array(np.where(np.array(recept)==59)[1])[0]]
del recept[0][np.array(np.where(np.array(recept)==64)[1])[0]]
del recept[0][np.array(np.where(np.array(recept)==74)[1])[0]]
del recept[0][np.array(np.where(np.array(recept)==80)[1])[0]]
del recept[0][np.array(np.where(np.array(recept)==44)[1])[0]]
del recept[0][np.array(np.where(np.array(recept)==48)[1])[0]]
del recept[0][np.array(np.where(np.array(recept)==35)[1])[0]]
del recept[0][np.array(np.where(np.array(recept)==40)[1])[0]]
recept=np.array(recept)

accmean = np.delete(accmean,np.array(np.where(np.array(recept)==20)[1])[0])
accmean = np.delete(accmean,np.array(np.where(np.array(recept)==59)[1])[0])
accmean = np.delete(accmean,np.array(np.where(np.array(recept)==64)[1])[0])
accmean = np.delete(accmean,np.array(np.where(np.array(recept)==74)[1])[0])
accmean = np.delete(accmean,np.array(np.where(np.array(recept)==80)[1])[0])
accmean = np.delete(accmean,np.array(np.where(np.array(recept)==44)[1])[0])
accmean = np.delete(accmean,np.array(np.where(np.array(recept)==48)[1])[0])
accmean = np.delete(accmean,np.array(np.where(np.array(recept)==35)[1])[0])
accmean = np.delete(accmean,np.array(np.where(np.array(recept)==40)[1])[0])

accmean25 = np.delete(accmean25,np.array(np.where(np.array(recept)==20)[1])[0])
accmean25 = np.delete(accmean25,np.array(np.where(np.array(recept)==59)[1])[0])
accmean25 = np.delete(accmean25,np.array(np.where(np.array(recept)==64)[1])[0])
accmean25 = np.delete(accmean25,np.array(np.where(np.array(recept)==74)[1])[0])
accmean25 = np.delete(accmean25,np.array(np.where(np.array(recept)==80)[1])[0])
accmean25 = np.delete(accmean25,np.array(np.where(np.array(recept)==44)[1])[0])
accmean25 = np.delete(accmean25,np.array(np.where(np.array(recept)==48)[1])[0])
accmean25 = np.delete(accmean25,np.array(np.where(np.array(recept)==35)[1])[0])
accmean25 = np.delete(accmean25,np.array(np.where(np.array(recept)==40)[1])[0])

accmean35 = np.delete(accmean35,np.array(np.where(np.array(recept)==20)[1])[0])
accmean35 = np.delete(accmean35,np.array(np.where(np.array(recept)==59)[1])[0])
accmean35 = np.delete(accmean35,np.array(np.where(np.array(recept)==64)[1])[0])
accmean35 = np.delete(accmean35,np.array(np.where(np.array(recept)==74)[1])[0])
accmean35 = np.delete(accmean35,np.array(np.where(np.array(recept)==80)[1])[0])
accmean35 = np.delete(accmean35,np.array(np.where(np.array(recept)==44)[1])[0])
accmean35 = np.delete(accmean35,np.array(np.where(np.array(recept)==48)[1])[0])
accmean35 = np.delete(accmean35,np.array(np.where(np.array(recept)==35)[1])[0])
accmean35 = np.delete(accmean35,np.array(np.where(np.array(recept)==40)[1])[0])

accmean5 = np.delete(accmean5,np.array(np.where(np.array(recept)==20)[1])[0])
accmean5 = np.delete(accmean5,np.array(np.where(np.array(recept)==59)[1])[0])
accmean5 = np.delete(accmean5,np.array(np.where(np.array(recept)==64)[1])[0])
accmean5 = np.delete(accmean5,np.array(np.where(np.array(recept)==74)[1])[0])
accmean5 = np.delete(accmean5,np.array(np.where(np.array(recept)==80)[1])[0])
accmean5 = np.delete(accmean5,np.array(np.where(np.array(recept)==44)[1])[0])
accmean5 = np.delete(accmean5,np.array(np.where(np.array(recept)==48)[1])[0])
accmean5 = np.delete(accmean5,np.array(np.where(np.array(recept)==35)[1])[0])
accmean5 = np.delete(accmean5,np.array(np.where(np.array(recept)==40)[1])[0])

err = np.delete(err,np.array(np.where(np.array(recept)==20)[1])[0])
err = np.delete(err,np.array(np.where(np.array(recept)==59)[1])[0])
err = np.delete(err,np.array(np.where(np.array(recept)==64)[1])[0])
err = np.delete(err,np.array(np.where(np.array(recept)==74)[1])[0])
err = np.delete(err,np.array(np.where(np.array(recept)==80)[1])[0])
err = np.delete(err,np.array(np.where(np.array(recept)==44)[1])[0])
err = np.delete(err,np.array(np.where(np.array(recept)==48)[1])[0])
err = np.delete(err,np.array(np.where(np.array(recept)==35)[1])[0])
err = np.delete(err,np.array(np.where(np.array(recept)==40)[1])[0])

err25 = np.delete(err25,np.array(np.where(np.array(recept)==20)[1])[0])
err25 = np.delete(err25,np.array(np.where(np.array(recept)==59)[1])[0])
err25 = np.delete(err25,np.array(np.where(np.array(recept)==64)[1])[0])
err25 = np.delete(err25,np.array(np.where(np.array(recept)==74)[1])[0])
err25 = np.delete(err25,np.array(np.where(np.array(recept)==80)[1])[0])
err25 = np.delete(err25,np.array(np.where(np.array(recept)==44)[1])[0])
err25 = np.delete(err25,np.array(np.where(np.array(recept)==48)[1])[0])
err25 = np.delete(err25,np.array(np.where(np.array(recept)==35)[1])[0])
err25 = np.delete(err25,np.array(np.where(np.array(recept)==40)[1])[0])

print(err35)

err35 = np.delete(err35,np.array(np.where(np.array(recept)==20)[1])[0])
err35 = np.delete(err35,np.array(np.where(np.array(recept)==59)[1])[0])
err35 = np.delete(err35,np.array(np.where(np.array(recept)==64)[1])[0])
err35 = np.delete(err35,np.array(np.where(np.array(recept)==74)[1])[0])
err35 = np.delete(err35,np.array(np.where(np.array(recept)==80)[1])[0])
err35 = np.delete(err35,np.array(np.where(np.array(recept)==44)[1])[0])
err35 = np.delete(err35,np.array(np.where(np.array(recept)==48)[1])[0])
err35 = np.delete(err35,np.array(np.where(np.array(recept)==35)[1])[0])
err35 = np.delete(err35,np.array(np.where(np.array(recept)==40)[1])[0])

err5 = np.delete(err5,np.array(np.where(np.array(recept)==20)[1])[0])
err5 = np.delete(err5,np.array(np.where(np.array(recept)==59)[1])[0])
err5 = np.delete(err5,np.array(np.where(np.array(recept)==64)[1])[0])
err5 = np.delete(err5,np.array(np.where(np.array(recept)==74)[1])[0])
err5 = np.delete(err5,np.array(np.where(np.array(recept)==80)[1])[0])
err5 = np.delete(err5,np.array(np.where(np.array(recept)==44)[1])[0])
err5 = np.delete(err5,np.array(np.where(np.array(recept)==48)[1])[0])
err5 = np.delete(err5,np.array(np.where(np.array(recept)==35)[1])[0])
err5 = np.delete(err5,np.array(np.where(np.array(recept)==40)[1])[0])

#plt.figure(figsize=(13,8))
#plt.errorbar(recept, accmean,yerr = err, color = 'black', marker = 'x', linestyle=' ', label = 'Harsh Accuracy')
#plt.errorbar(recept, accmean25,yerr=err25,marker='o', markersize=5,c = 'tomato', mfc = 'red',linestyle='-', label='NN Cap Size = 0.25')
#plt.errorbar(recept, accmean35,yerr=err35,markersize=5, c= 'lightgreen', mfc = 'green', linestyle='-',label='NN Cap Size = 0.35')
#plt.errorbar(recept, accmean5,yerr = err5,markersize=5, c = 'skyblue', mfc = 'blue', linestyle='-', label='NN Cap Size = 0.50')
plt.plot(recept[0,:], accmean, color = 'purple', marker = 'o',markersize=13, linestyle='-',linewidth=2, label = 'Harsh accuracy')
plt.fill_between(recept[0,:], accmean-err, accmean+err, edgecolor = 'purple', facecolor='#D2B4DE')
plt.plot(recept[0,:], accmean25,marker='o', markersize=13,c = 'red', mfc = 'red',linestyle='-',linewidth=2, label='n.n.a. cap ratio = 0.25')
plt.fill_between(recept[0,:], accmean25-err25, accmean25+err25,edgecolor = 'red', facecolor='#F5B7B1')
plt.plot(recept[0,:], accmean35,marker='o',markersize=13, c= 'green', mfc = 'green', linestyle='-',linewidth=2,label='n.n.a. cap ratio = 0.35')
plt.fill_between(recept[0,:], accmean35-err35, accmean35+err35,edgecolor = 'green', facecolor='#A9DFBF')
plt.plot(recept[0,:], accmean5,marker='o',markersize=13, c = 'blue', linestyle='-', linewidth=2,label='n.n.a. cap ratio = 0.50')
plt.fill_between(recept[0,:], accmean5-err5, accmean35+err5,edgecolor = 'blue', facecolor='#A9CCE3')
plt.legend()
plt.xticks(np.arange(5,100,15))
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of receptors')
plt.savefig('recept_NNerroract.png')
"""
plt.rcParams.update({'font.size': 35})

diffusion = np.arange(0.1,2.1,0.1)
plt.figure(figsize=(22,14))
accuracy = read_datafile('update_accuracy_total_diff19-0.1')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
acc25 = read_datafile('25update_accuracy_total_diff19-0.1')
del acc25[0]
accmean25 = np.mean(acc25, axis =0)
err25= np.std(accuracy, axis=0)
acc35 = read_datafile('35update_accuracy_total_diff19-0.1')
del acc35[0]
accmean35 = np.mean(acc35, axis =0)
err35= np.std(accmean35, axis=0)
acc5 = read_datafile('5update_accuracy_total_diff19-0.1')
del acc5[0]
accmean5 = np.mean(acc5, axis =0)
err5= np.std(accmean5, axis=0)
plt.plot(diffusion, accmean, color = 'purple', marker = 'o',markersize=13, linestyle='-',linewidth=2, label = 'Harsh Accuracy')
plt.fill_between(diffusion, accmean-err, accmean+err, edgecolor = 'purple', facecolor='#D2B4DE')
plt.plot(diffusion, accmean25,marker='o', markersize=13,c = 'red', mfc = 'red',linestyle='-',linewidth=2, label='NN Cap Size = 0.25')
plt.fill_between(diffusion, accmean25-err25, accmean25+err25,edgecolor = 'red', facecolor='#F5B7B1')
plt.plot(diffusion, accmean35,marker='o',markersize=13, c= 'green', mfc = 'green', linestyle='-',linewidth=2,label='NN Cap Size = 0.35')
plt.fill_between(diffusion, accmean35-err35, accmean35+err35,edgecolor = 'green', facecolor='#A9DFBF')
plt.plot(diffusion, accmean5,marker='o',markersize=13, c = 'blue', linestyle='-', linewidth=2,label='NN Cap Size = 0.50')
plt.fill_between(diffusion, accmean5-err5, accmean35+err5,edgecolor = 'blue', facecolor='#A9CCE3')
plt.legend()
plt.xticks(np.arange(0.1,2.1,0.3))
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Diffusion constant')
plt.savefig('diffusion_NNerror.png')
"""

"""
plt.rcParams.update({'font.size': 35})

distance= [2,3,4,5,6,7,8,9,10]
accuracy = read_datafile('accuracy_total_distances19-0.1')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
acc25 = read_datafile('accuracynn0.25_total_distances19-0.1')
del acc25[0]
accmean25 = np.mean(acc25, axis =0)
err25= np.std(acc25, axis=0)
acc35 = read_datafile('accuracynn0.35_total_distances19-0.1')
del acc35[0]
accmean35 = np.mean(acc35, axis =0)
err35= np.std(accmean35, axis=0)
acc5 = read_datafile('accuracynn0.50_total_distances19-0.1')
del acc5[0]
accmean5 = np.mean(acc5, axis =0)
err5= np.std(accmean5, axis=0)
plt.figure(figsize=(22,14))
plt.plot(distance, accmean, color = 'purple', marker = 'o', markersize=13, linestyle='-', linewidth=2, label = 'Harsh accuracy')
plt.fill_between(distance, accmean-err, accmean+err, edgecolor = 'purple', facecolor='#D2B4DE')
plt.plot(distance, accmean25,marker='o', markersize=13,c = 'red', mfc = 'red',linestyle='-',linewidth=2, label='n.n.a. cap ratio= 0.25')
plt.fill_between(distance, accmean25-err25, accmean25+err25,edgecolor = 'red', facecolor='#F5B7B1')
plt.plot(distance, accmean35,marker='o',markersize=13, c= 'green', mfc = 'green', linestyle='-',linewidth=2,label='n.n.a. cap ratio = 0.35')
plt.fill_between(distance, accmean35-err35, accmean35+err35,edgecolor = 'green', facecolor='#A9DFBF')
plt.plot(distance, accmean5,marker='o',markersize=13, c = 'blue', linestyle='-',linewidth=2, label='n.n.a. cap ratio = 0.50')
plt.fill_between(distance, accmean5-err5, accmean35+err5,edgecolor = 'blue', facecolor='#A9CCE3')
plt.legend()
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Distance from cell center to source')
plt.savefig('distance19-0.1_NNerror.png')
"""
"""
"""
"""
plt.rcParams.update({'font.size': 35})
nodes = np.arange(3,30)
accuracy = read_datafile('accuracy_total_nodes_big_update')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
print(np.where(accmean == np.max(accmean)))
err= np.std(accuracy, axis=0)
plt.figure(figsize=(18,10))
print(err)
    #plt.figure()
plt.errorbar(nodes,accmean,yerr=err,elinewidth=2, linewidth = 2, marker = 'o', markersize= 13, color = 'red')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel('Number of nodes in hidden layer')
plt.savefig('nodeserr_update_big.png')
"""
"""
alpha= [0.0001,0.001,0.01,0.1,0.5,1.0,1.5,2.0,2.5,3.0]
accuracy = read_datafile('accuracy_total_alphas_16')
del accuracy[0]
accmean = np.mean(accuracy, axis =0)
err= np.std(accuracy, axis=0)
plt.figure(figsize=(13,8))
plt.errorbar(alpha,accmean,yerr=err,elinewidth=1, linewidth = 0, marker = 'o', markersize= 8, color = 'red')
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel(r'$\alpha$')
plt.savefig('alpha.png')
"""
"""
plt.rcParams.update({'font.size': 30})
alpha = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1]
plt.figure(figsize=(12,9))
accuracytest = read_datafile('accuracyalphas')
del accuracytest[0]
errt= np.std(accuracytest,axis =0)
acctest = np.mean(accuracytest,axis =0)

plt.errorbar(alpha, acctest, yerr= errt, marker='o',markersize=13, c = 'blue', linestyle='-',linewidth=2)
ax = plt.axes()
ax.set_xscale("log")
plt.ylabel('Accuracy of MLPClassifier')
plt.xlabel(r'$\alpha$ value')
plt.xticks([0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1])
plt.legend()
plt.savefig('alphas300.png')
"""