#!/bin/sh
## one day
#PBS -l walltime=100:00:00
#PBS -l mem=1900mb
#PBS -l ncpus=1
#PBS -e RnT_mips_extension.err
#PBS -o RnT_mips_extension.out

cd $PBS_O_WORKDIR


echo script $0
pwd
hostname
date



#time ./RnT_mips -s seed -r -g gamma -w velocity -n POTnu -N 2 -M maxT -m 0.01 -D diff -x POTxi -l length -t deltat > RnT_mips_extension.dat
time ./BrownianParticle_stndln -S seed -b boost -I iterations -w warmup -s source -m maxT > BrownianParticle_stndln_extension.dat
date
