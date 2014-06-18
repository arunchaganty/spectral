#!/bin/bash

K=2
D=3
SEED=1
ATTEMPTS=1000
for noise in 1e0 1e-1 5e-2 1e-2 5e-3 1e-3 5e-4 1e-4; do
  echo $noise
  python experiments/pairwise.py --k $K --d $D --seed $SEED --attempts $ATTEMPTS --noise $noise | tee -a noise_exp.txt
done;
