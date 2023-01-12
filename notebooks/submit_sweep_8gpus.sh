#!/bin/bash
algorithm_type="sweep_functor_onlyxflow_radial"
dataset="texas"
date="jan11"
for i in {0..8}
do 
	nohup python ${algorithm_type}_gpu${i}_${dataset}_${date}.py > ${dataset}_${date}_gpu${i}.log 2>&1 &
done

