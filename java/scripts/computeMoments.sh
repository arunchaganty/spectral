#!/bin/bash

for d in 10 50 80 100 150 200; do
  for i in `seq 1 10`; do
    echo ./run.sh learning.data.MomentComputer -dataPath data/wsj/wsj.words -mapPath data/wsj/wsj_index.words -execPoolDir out -randomProjDim 100 -randomProjSeed $((i+d))
  done;
done;

