#!/bin/sh
## one day
#PBS -l walltime=100:00:00
#PBS -l mem=1900mb
#PBS -l ncpus=1
#PBS -e RnT_mips_RUN01_P03_20220929_024848.err
#PBS -o RnT_mips_RUN01_P03_20220929_024848.out

cd $PBS_O_WORKDIR


echo script $0
pwd
hostname
date



#time ./RnT_mips -s 31053 -r -g gamma -w velocity -n POTnu -N 2 -M 1000. -m 0.01 -D diff -x POTxi -l length -t deltat > RnT_mips_RUN01_P03_20220929_024848.dat
time ./BrownianParticle_stndln -s 31053 -b 0.1 -I 10000 -w 100 -s 0,0,3. -m 1000. > BrownianParticle_stndln_RUN01_P03_20220929_024848.dat
date
