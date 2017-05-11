#!/bin/bash

. /home/UParse/torch/install/bin/torch-activate

while getopts i:o: opt; do
  case $opt in
  i)
      input=$OPTARG
      ;;
  o)
      output=$OPTARG
      ;;
  esac
done

shift $((OPTIND - 1))

# codedir=/home/UParse/parser
codedir=/home/claravania/Documents/projects/UParse
runType=dense_dist

echo $input
echo $output

python3 $codedir/scripts/parse.py $input $output $codedir $runType
