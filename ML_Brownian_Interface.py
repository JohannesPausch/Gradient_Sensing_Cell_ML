# communication between python ML and c-code BrownianParticle.c 

from subprocess import Popen, PIPE


def init_BrownianParticle(xpos,ypos,zpos,rate,seed=None):
    if seed==None:
        brownian_pipe = Popen(['./BrownianParticle.o'], shell=True, stdout=PIPE, stdin=PIPE)
    else:
        brownian_pipe = Popen(['./BrownianParticle.o -S '+str(seed)], shell=True, stdout=PIPE, stdin=PIPE)
    brownian_pipe.stdin.write(bytes(str(xpos)+','+str(ypos)+','+str(zpos)+'\n', 'UTF-8'))
    brownian_pipe.stdin.flush()
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split()
    received = [float(x) for x in received]
    return brownian_pipe,received

def update_BrownianParticle(brownian_pipe,xpos,ypos,zpos):
    brownian_pipe.stdin.write(bytes(str(xpos)+','+str(ypos)+','+str(zpos)+'\n', 'UTF-8'))
    brownian_pipe.stdin.flush()
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split()
    received = [float(x) for x in received]
    return received

def stop_BrownianParticle(brownian_pipe):
    brownian_pipe.stdin.write(bytes("Thanks for all the fish!\n", 'UTF-8'))
    brownian_pipe.stdin.flush()
    return
