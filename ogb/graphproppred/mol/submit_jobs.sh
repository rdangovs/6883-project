#!/bin/bash
for i in {0..11}
do
  sbatch "scripts/$i.sh"
done