#!/bin/bash
ls *f > list
while read forfile
do
cat $forfile >> aftan.fo
done < list
