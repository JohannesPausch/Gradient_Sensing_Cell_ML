# $Header: /home/ma/p/pruess/.cvsroot/soc_sierpinski/Makefile,v 1.23 2017/08/24 14:03:37 pruess Exp $
## standalone 
CC=cc
EXTRA_CFLAGS=-Wall
ARCHITECTURE_OPTIM=
#OPTIM=-O3 -mcpu=G4 -mpowerpc
OPTIM=-O3
#OPTIM= -ggdb
OPTIM_LINK=-O3
OPTIM_LINK=
CFLAGS=-Wall
EXTRA_CFLAGS=
LIB=-lm
#
# debug
#CC=cc
#EXTRA_CFLAGS=-Wall -ggdb
#ARCHITECTURE_OPTIM=
#OPTIM=
#OPTIM_LINK=
#CFLAGS=-Wall -ggdb
#LIB=-lm
##
## tbird 
##CC=cc
#EXTRA_CFLAGS=-Wall
#OPTIM=-O5
#OPTIM_LINK=-O3
#
# anantham 
#CC=mpicc
#EXTRA_CFLAGS=-Wall
#OPTIM=-O5
#OPTIM_LINK=-O3
#
# System X
#CC=mpixlc
#EXTRA_CFLAGS=-DSYSTEM_X
#OPTIM=-O5
#OPTIM_LINK=-O3


programs=BrownianParticle

all: ${programs}

clean:
	rm -f *.o ${programs} 

BrownianParticle_OBJS = BrownianParticle.o
BrownianParticle: ${BrownianParticle_OBJS} Gradient_Sensing_Cell_ML_git_stamps.h
	${CC} ${CFLAGS} ${OPTIM_LINK} ${EXTRA_CFLAGS} ${CONF_FLAGS} -o $@ ${BrownianParticle_OBJS} ${LIB} -lgsl

BrownianParticle_fifo_OBJS = BrownianParticle_fifo.o
BrownianParticle_fifo: ${BrownianParticle_fifo_OBJS} Gradient_Sensing_Cell_ML_git_stamps.h
	${CC} ${CFLAGS} ${OPTIM_LINK} ${EXTRA_CFLAGS} ${CONF_FLAGS} -o $@ ${BrownianParticle_fifo_OBJS} ${LIB} -lgsl

BrownianParticle_stndln_OBJS = BrownianParticle_stndln.o
BrownianParticle_stndln: ${BrownianParticle_stndln_OBJS} 
	${CC} ${CFLAGS} ${OPTIM_LINK} ${EXTRA_CFLAGS} ${CONF_FLAGS} -o $@ ${BrownianParticle_stndln_OBJS} ${LIB} -lgsl

Gradient_Sensing_Cell_ML_git_stamps.h: .
	./git_stamps.sh > Gradient_Sensing_Cell_ML_git_stamps.h

.SUFFIXES : .o .c .h
.c.o :
	${CC} ${OPTIM} ${CFLAGS} ${EXTRA_CFLAGS} ${CONF_FLAGS} -c $<

