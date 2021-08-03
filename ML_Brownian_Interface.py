# communication between python ML and c-code BrownianParticle.c 

from subprocess import Popen, PIPE
separator = ' '

def init_BrownianParticle(distance=None,rate=None,diffusion=None,seed=None,cutoff=None,events=None,training=None,iterations=None):
    command = './BrownianParticle.o'
    if distance != None:
        command += ' -d '+str(distance)
    if rate != None:
        command += ' -r '+str(rate)
    if seed != None:
        command += ' -S '+str(seed)
    if cutoff != None:
        command += ' -c '+str(cutoff)
    if events != None:
        command += ' -E '+str(events)
    if iterations != None:
        command += ' -N '+str(iterations)
    if diffusion != None:
        command += ' -s '+str(diffusion)
    if training != None:
        command += ' -t y '
    brownian_pipe = Popen([command], shell=True, stdout=PIPE, stdin=PIPE)
    brownian_pipe.stdin.write(bytes(str(xpos)+separator+str(ypos)+separator+str(zpos)+'\n', 'UTF-8'))
    brownian_pipe.stdin.flush()
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    received = [float(x) for x in received]
    return brownian_pipe,received

def update_BrownianParticle(brownian_pipe,step_theta=None,step_phi=None):
    if step_theta != None and step_phi != None:
        brownian_pipe.stdin.write(bytes(str(step_theta)+separator+str(step_phi)+'\n', 'UTF-8'))
    else:
        brownian_pipe.stdin.write(bytes('\n', 'UTF-8'))
    brownian_pipe.stdin.flush()
    received = brownian_pipe.stdout.readline().strip().decode('ascii').split(separator)
    if received[0]=='SOURCE':
        return 'SOURCE FOUND'
    received = [float(x) for x in received]
    return received

def stop_BrownianParticle(brownian_pipe):
    brownian_pipe.stdin.write(bytes("Thanks for all the fish!\n", 'UTF-8'))
    brownian_pipe.stdin.flush()
    return
