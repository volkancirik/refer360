#!/usr/bin/env sh

list=$1
completed=0
all=`wc -l $list | cut -d" " -f1`
while read line
do

    echo $line
    eval $line

    completed=`expr $completed + 1`
    echo "DONE!!! $completed|$all"
    echo "DONE!!! $completed|$all" > $list.log
done < $list
