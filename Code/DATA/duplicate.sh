#!/bin/sh
# ./duplicate.sh RUN10 copies all relevant files from RUN10 to the next available slot, ready to be edited.

last=`ls -d RUN* | sort | tail -1`
if [ $# -le 0 ]
then
  oldrun=${last}
else
  oldrun=$1
fi

echo oldrun: $oldrun

if [ ! -d $oldrun ]
then
  echo oldrun $oldrun is not a directory
  exit
fi


lastno=`echo $last | sed 's/RUN0*//'`

echo lastno: $lastno
number=$(($lastno + 1))


#echo $lastno $number
newrun=RUN`printf %02i $number`

echo $newrun

#newrun="RUN12"
if [ -e $newrun ]
then
  echo $newrun exists
  exit
fi

mkdir $newrun
date >> ${oldrun}/readme.txt
echo "Duplicated $oldrun to $newrun." >> ${oldrun}/readme.txt
for f in run.txt readme.txt
do
  cp $oldrun/$f $newrun
done

for f in run.sh Makefile BrownianParticle_magic.h BrownianParticle_stndln.c qsub_template.sh
do
  cp $f $newrun
done

echo Compiling

date >> ${newrun}/readme.txt 2>&1
echo "Compiling " >> ${newrun}/readme.txt 2>&1
cd $newrun
make BrownianParticle_stndln >> readme.txt 2>&1
cd ..

