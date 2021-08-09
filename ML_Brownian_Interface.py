# communication between python ML and c-code BrownianParticle.c 

from subprocess import Popen, PIPE
from scipy.stats import special_ortho_group
import numpy as np
separator = ' '

def init_BrownianParticle(distance=None,rate=None,diffusion=None,radius=1,use_seed=None,cutoff=None,events=None,training=None,iterations=None):
    command = './BrownianParticle_fifo.o'
    if use_seed != None:
        command += ' -S '+str(use_seed)
        np.random.seed(seed=use_seed)
    if distance != None:
        y=np.matmul(special_ortho_group.rvs(3),np.array([distance,0,0]))
        command += ' -s '+str(y[0])+separator+str(y[1])+separator+str(y[2])
    if rate != None:
        command += ' -r '+str(rate)
    if cutoff != None:
        command += ' -c '+str(cutoff)
    if events != None:
        command += ' -E '+str(events)
    if iterations != None:
        command += ' -N '+str(iterations)
    if diffusion != None:
        command += ' -d '+str(diffusion)
    if training != None:
        command += ' -t y '
    if radius != 1:
        command += ' -rad '+str(radius)
    print('used command: '+command)
    brownian_pipe = Popen([command], shell=True, stdout=PIPE, stdin=PIPE)
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    while received[0] == '#':
        received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
        print(received)
    try: 
        received = [float(x) for x in received]
        print(received)
    except:
        print('not a float')
        print(received)
    return brownian_pipe,received

def init_BrownianParticle_test(distance=None,rate=None,diffusion=None,use_seed=None,cutoff=None,events=None,training=None,iterations=None):
    return 1,np.random.rand(5)

def update_BrownianParticle(brownian_pipe,step_theta=None,step_phi=None):
    if step_theta != None and step_phi != None:
        brownian_pipe.stdin.write(bytes(str(np.cos(step_phi)*np.sin(step_theta))+separator+str(np.sin(step_phi)*np.sin(step_theta))+separator+str(np.cos(step_theta))+'\n', 'UTF-8'))
    else:
        brownian_pipe.stdin.write(bytes('0'+separator+'0'+separator+'0\n', 'UTF-8'))
    brownian_pipe.stdin.flush()
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    if received[0]=='#' and received[1] == 'HEUREKA':
        return 'SOURCE FOUND'
    while received[0] == '#':
        received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    try: 
        received = [float(x) for x in received]
    except:
        print('not a float')
        print(received)
    return received

def update_BrownianParticle_test(brownian_pipe,step_theta=None,step_phi=None):
    return np.random.rand(5)

def stop_BrownianParticle(brownian_pipe):
    brownian_pipe.stdin.write(bytes("Thanks for all the fish!\n", 'UTF-8'))
    brownian_pipe.stdin.flush()
    return
