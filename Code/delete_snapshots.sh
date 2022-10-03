#!/bin/sh




for n in 0 1 2 3 4 5 6 7 8 9 
do
  echo $n
  rm -f BrownianParticle_stndln_snapshot*${n}.txt
done
