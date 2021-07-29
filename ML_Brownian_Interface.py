# communication between python ML and c-code BrownianParticle.c 

from subprocess import Popen, PIPE
separator = ' '

def init_BrownianParticle(xpos,ypos,zpos,rate,seed=None,cutoff=None,events=None,iterations=None,sigma=None):
    command = './BrownianParticle.o'
    if seed != None:
        command += ' -S '+str(seed)
    if cutoff != None:
        command += ' -c '+str(cutoff)
    if events != None:
        command += ' -E '+str(events)
    if iterations != None:
        command += ' -N '+str(iterations)
    if sigma != None:
        command += ' -s '+str(sigma)
    brownian_pipe = Popen([command], shell=True, stdout=PIPE, stdin=PIPE)
    brownian_pipe.stdin.write(bytes(str(xpos)+separator+str(ypos)+separator+str(zpos)+'\n', 'UTF-8'))
    brownian_pipe.stdin.flush()
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    received = [float(x) for x in received]
    return brownian_pipe,received

def update_BrownianParticle(brownian_pipe,xpos,ypos,zpos):
    brownian_pipe.stdin.write(bytes(str(xpos)+separator+str(ypos)+separator+str(zpos)+'\n', 'UTF-8'))
    brownian_pipe.stdin.flush()
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    received = [float(x) for x in received]
    return received

def stop_BrownianParticle(brownian_pipe):
    brownian_pipe.stdin.write(bytes("Thanks for all the fish!\n", 'UTF-8'))
    brownian_pipe.stdin.flush()
    return
