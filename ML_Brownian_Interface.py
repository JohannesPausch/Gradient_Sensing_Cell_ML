# communication between python ML and c-code BrownianParticle.c 

from os import error
from subprocess import Popen, PIPE
from scipy.stats import special_ortho_group
import numpy as np
separator = ' '

def init_BrownianParticle(xpos=None,ypos=None,zpos=None,rate=None,diffusion=None,radius=1,use_seed=None,cutoff=None,events=None,iterations=None):
    command = './BrownianParticle_fifo.o'
    if use_seed != None:
        command += ' -S '+str(use_seed)
        np.random.seed(seed=use_seed)
    if xpos != None:
        command += ' -s '+str(xpos)+','+str(ypos)+','+str(zpos)
    else:
        if cutoff != None:
            y=np.matmul(special_ortho_group.rvs(3),np.array([radius+(cutoff-radius)*np.random.rand(1),0,0]))
            command += ' -s '+str(y[0])+','+str(y[1])+','+str(y[2])
        else:
            y=np.matmul(special_ortho_group.rvs(3),np.array([radius+radius*np.random.rand(1),0,0]))
            command += ' -s '+str(y[0])+','+str(y[1])+','+str(y[2])
    if rate != None:
        command += ' -R '+str(rate)
    if cutoff != None:
        command += ' -c '+str(cutoff)
    if events != None:
        command += ' -E '+str(events)
    if iterations != None:
        command += ' -N '+str(iterations)
    if diffusion != None:
        command += ' -d '+str(diffusion)
    if radius != 1:
        command += ' -r '+str(radius)
    print('used command: '+command)
    brownian_pipe = Popen([command], shell=True, stdout=PIPE, stdin=PIPE)
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    while received[0] == '#':
        received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    try: 
        received = [float(x) for x in received]
    except:
        print('Error: not a float')
        print(received)
    radius = np.sqrt(xpos*xpos+ypos*ypos+zpos*zpos)
    source_theta,source_phi=np.arccos(zpos/radius), np.mod(np.arctan2(ypos,xpos),2*np.pi)
    return brownian_pipe,received,[source_theta,source_phi]

def init_BrownianParticle_test(distance=None,rate=None,diffusion=None,use_seed=None,cutoff=None,events=None,training=None,iterations=None):
    return 1,np.random.rand(3)

def stop_BrownianParticle(brownian_pipe):
    brownian_pipe.stdin.write(bytes("Thanks for all the fish!\n", 'UTF-8'))
    brownian_pipe.stdin.flush()
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    return 'C-PRORGAM STOPPED'

def update_BrownianParticle(brownian_pipe,step_theta=None,step_phi=None):
    if step_theta != None and step_phi != None:
        brownian_pipe.stdin.write(bytes(str(np.cos(step_phi)*np.sin(step_theta))+separator+str(np.sin(step_phi)*np.sin(step_theta))+separator+str(np.cos(step_theta))+'\n', 'UTF-8'))
    else:
        brownian_pipe.stdin.write(bytes('0'+separator+'0'+separator+'0\n', 'UTF-8'))
    brownian_pipe.stdin.flush()
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    if received[0] == 'Error:':
        error_string = received[0]
        for x in received:
            error_string += x
        print(error_string)
        return np.array([0,0,0])
    elif received[3] == 'source':
        source_x,source_y,source_z = [float(received[5]),float(received[6]),float(received[7])]
        radius = np.sqrt(source_x*source_x+source_y*source_y+source_z*source_z)
        source_theta,source_phi=np.arccos(source_z/radius), np.mod(np.arctan2(source_y,source_x),2*np.pi)
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    if received[0]=='HEUREKA!':
        stop_BrownianParticle(brownian_pipe)
        return ['SOURCE', 'FOUND']
    while received[0] == '#':
        received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    try: 
        received = [float(x) for x in received]
    except:
        print('not a float')
        error_string = received[0]
        for x in received:
            error_string += x
        print(error_string)
    return received,[source_theta,source_phi]

def update_BrownianParticle_test(brownian_pipe,step_theta=None,step_phi=None):
    return np.random.rand(3)


