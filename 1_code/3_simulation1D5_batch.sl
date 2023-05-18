#!/bin/bash

for evaluationEstimator in {'IPW','sIPW','gcomp','AIPWE'}
do
	for nObs in {200,500,1000}
	do
		for w in {0,1,2}
		do
		
		sbatch "3_simulation1D5.sl" $evaluationEstimator $nObs $w

		done
	done
done	
