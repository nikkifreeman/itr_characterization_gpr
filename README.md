# ITR Characterization via surrogate for the value function

## Directories
There are 3 directories in this repo

  * 1_code,
  * 2_pipeline, and
  * 3_output.
  
Code for the simulations and data analysis are in 1_code. Intermediate outputs from the code that are used later in the analyses are in  2_pipeline. Outputs for the manuscript are contained in 3_output. 

## Prefixes
Shared prefixes indicates that code belongs to the same analysis, e.g. 1_code/3_simulation1D4.py generates 2_pipeline/3_simulation1D4_results. 

  * The prefix 3_simulation1D5 corresponds to Simulation Setting 1.
  * The prefix 3_simulation1D3 corresponds to Simulation Setting 2.
  * The prefix 3_simulation1D4 corresponds to Simulation Setting 3.
  * The prefix 10_ corresponds to the data analysis.
  
## Pipeline
### Simulation 
The simulations were conducted at UNC during the Fall of 2022 using Longleaf on the high performance computing cluster. To generate the ground truth values for all of the ITRs in the class under examination, use the code

  * 3_simulation1D5_trueValues.ipynb for Simulation Setting 1,
  * 3_simulation1D3_trueValues.ipynb for Simulation Setting 2, and
  * 3_simulation1D4_trueValues.ipynb for Simulation Setting 3.

The ground truth values are written to 

  * 3_simulation1D5_*_trueValues.csv for Simulation Setting 1,
  * 3_simulation1D3_*_trueValues.csv for Simulation Setting 2, and
  * 3_simulation1D4_*_trueValues.csv for Simulation Setting 3,
  
where * denotes either 0.75, 1, or 1.25 which are parameters in the simulation.

The simulation runs were completed in batch mode. 

  * 1_code/3_simulation1D*_batch.sl is the script for batch mode. It sets the simulations settings, i.e., number of observations, value for $w$, and estimation strategy.
  * The simulation parameters are then passed to 1_code/3_simulation1D*.sl which is the slurm file that calls the python code file. It passes the simulation parameter settings to the python code file.
  * 1_code/3_simulation1D*.py is the python file contains the simulation code. It receives the simulation parameters and implements the specific simulation analysis.
  * The results from each simulation are contained in the directories 2_pipeline/3_simulation1D_results. The specific file names have the convention prefix + [underscore] + estimator + [underscore] + number of observations + [underscore] + w setting + ".csv"
  * The results were analyzed and synthesized in 1_code/3_simulation1D*_analysis.Rmd.
  * Figures for the manuscript are stored in 3_output.

Note that * denotes either 5 (Setting 1), 3 (Setting 2), or 4 (Setting 1).

### Data analysis

The data analysis was conducted using 1_code/10_realDataAnaysis.ipynb, and the corresponding midstream outputs are 2_pipeline/10_gpValueSurrogateOverTheta.csv, 2_pipeline/10_optimalTheta.csv, and 2_pipeline/10_policiesExplored.csv. Synthesis of results from the analysis as well as figures were generated using 10_realDataAnalysis.Rmd, and the corresponding figures are contained in 3_output.



  
  
