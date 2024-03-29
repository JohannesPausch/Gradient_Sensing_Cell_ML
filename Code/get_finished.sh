#!/bin/sh


for f
do
  for key in FINISHED
  do
    extension=`echo $key | tr '[:upper:]' '[:lower:]'`
    file=`echo $f | sed 's/\(.*\)\..../\1/'`
    echo '#' $key $file $extension
    cat $f | grep ${key} | sed 's/.*'${key}' //' > ${file}.txt_${extension}
    # m1 is time
    # n1 is nudges
    cat ${file}.txt_${extension} | awk ' { t++; if ($2==4) {m0++; m1+=$6; n1+=$7 } } END { print t, m0, m1/m0, n1/m0; } '
  done
done
