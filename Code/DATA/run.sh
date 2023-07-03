#!/bin/sh

pwd=`pwd`
run=`basename $pwd`


seed=10007
seed=`echo 'echo $RANDOM' | bash`
#seed=$(( $seed \* 2 + 1 ))
seed=$(( $seed * 2 + 1 ))
echo base seed $seed
dt=`date +%Y%m%d_%H%M%S`

grep -v ^# run.txt | sed 's/#.*//' | sed 's/^[ 	]*//' | grep -v '^$' | while read parallels boost iterations warmup source maxT gobble
do
  for p in `seq 1 $parallels`
  do
  echo parallels $p of $parallels


    seed=$(( $seed + 2 ))
    pp=`printf %02i $p`

    #extension=${run}_${gamma}_${w}_${nu}_${maxT}_${diff}_${xi}_${length}_${deltat}_P${pp}_${dt}
    extension=${run}_P${pp}_${dt}
    qsub_file=qsub${extension}.sh
    cat qsub_template.sh  | sed 's/boost/'${boost}'/g' \
                          | sed 's/iterations/'${iterations}'/g'  \
                          | sed 's/warmup/'${warmup}'/g'  \
                          | sed 's/source/'${source}'/g'  \
                          | sed 's/maxT/'${maxT}'/g'  \
                          | sed 's/seed/'${seed}'/g'  \
                          | sed 's/extension/'${extension}'/g'  \
                          | sed 's/parallel/'${pp}'/g'  > $qsub_file
    chmod a+x $qsub_file
    echo qsub $qsub_file
    qsub $qsub_file
    #time ./BrownianParticle_stndln -s seed -b boost -I iterations -w warmup -s source -m maxT > BrownianParticle_stndln_extension.dat


done
done