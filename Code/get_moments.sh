#!/bin/sh


for f
do
  for key in DISTANCE INTERARRIVAL
  do
    extension=`echo $key | tr '[:upper:]' '[:lower:]'`
    file=`echo $f | sed 's/\(.*\)\..../\1/'`
    echo $key $file $extension
    cat $f | grep MOM_${key} | sed 's/.*'${key}' //' > ${file}.txt_${extension}
  done
done
