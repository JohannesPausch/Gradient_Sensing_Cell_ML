#!/bin/sh

for f 
do 
   echo Doing $f
   grep -m 1 velocity  $f | sed 's/.* is \(.*\) distance.*/\1/' > ${f}_first_velocity
   grep 'FINIS\|veloci' $f | grep -A 1 FINISH | grep velocity | sed 's/.* is \(.*\) distance.*/\1/' >> ${f}_first_velocity
done
