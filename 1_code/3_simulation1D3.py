# 3_simulation1D1
import pandas as pd
import numpy as np
import sys
import random
from sklearn.linear_model import LinearRegression
import sklearn as sk
from skopt import gp_minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Matern

print(sys.argv)

# Functions --------------------------------------------------------------------
# Generate observed data
def generateObservedData(sampleSize, w, gamma):
    intercept = np.ones(sampleSize)
    X = w*np.random.uniform(0, 1, sampleSize)
    A = random.choices([0, 1], weights = [0.5, 0.5], k = sampleSize)
    Y = gamma[0]*intercept + gamma[1]*X + A*np.cos(X*2*np.pi)
    return intercept, X, A, Y
    
# IPW estimate (negated)
def computeIPW(beta, obsData, n):
    obsData['A_d'] = np.logical_or(obsData['X'] < beta[0], obsData['X'] > beta[1])*1.0
    obsData['C_d'] = np.where(obsData['A'] == obsData['A_d'], 1, 0)
    obsData['pi_d'] = 0.5
    obsData['summand'] = obsData['C_d']*obsData['Y']/obsData['pi_d']
    
    # Estimate the value
    vhat_ipw = sum(obsData['summand'])/(n)
    
    return -1*vhat_ipw

# Stabilized IPW estimate (negated)
def computeStabilizedIPW(beta, obsData, n):
    obsData['A_d'] = np.logical_or(obsData['X'] < beta[0], obsData['X'] > beta[1])*1.0
    obsData['C_d'] = np.where(obsData['A'] == obsData['A_d'], 1, 0)
    obsData['pi_d'] = 0.5
    obsData['summand'] = obsData['C_d']*obsData['Y']*obsData['pi_d']
    
    # Estimate the value
    vhat_stabilizedIPW = sum(obsData['summand'])/(sum(obsData['C_d']*obsData['pi_d']))
    
    return -1*vhat_stabilizedIPW

# Regression estimator (G-computation) (negated)
def computeRegEst(beta, obsData, n):
    # X contains the covariates
    X = obsData.loc[:,['X', 'A']]
    X['int'] = np.ones(n)
    X['AX'] = np.multiply(obsData['X'], obsData['A'])
    X = X.loc[:, ['int', 'X', 'A', 'AX']]
    
    # Y contains the outcomes
    Y = obsData['Y']
    
    # Fit the regression model
    model = LinearRegression().fit(X, Y)
    
    # Calculate Qhat(H, 1)
    X_1 = X.copy(deep = True)
    X_1['A'] = 1
    Qhat1 = model.predict(X_1)
    
    # Calculate Qhat(H, 0)
    X_0 = X.copy(deep = True)
    X_0['A'] = 0
    Qhat0 = model.predict(X_0)
    
    # Calculate the treatment recommendation under the policy indexed by beta
    A_d = np.logical_or(obsData['X'] < beta[0], obsData['X'] > beta[1])*1.0
    
    # Estimate the value
    vhat = np.sum(np.where(A_d == 1, Qhat1, 0) + np.where(A_d == 0, Qhat0, 0))*(1/n)
    
    return -1*vhat

# AIPWE (negated)
def computeAIPWE(beta, obsData, n):
    # IPW piece
    obsData['A_d'] = np.logical_or(obsData['X'] < beta[0], obsData['X'] > beta[1])*1.0
    obsData['C_d'] = np.where(obsData['A'] == obsData['A_d'], 1, 0)
    obsData['pi_d'] = 0.5
#     obsData['summand'] = obsData['C_d']*obsData['Y']*obsData['pi_d']
#     obsData['weight'] = (obsData['C_d'] - obsData['pi_d'])/obsData['pi_d']
    
    # Regression piece
    # X contains the covariates
    X = obsData.loc[:,['X', 'A']]
    X['int'] = np.ones(n)
    X['AX'] = np.multiply(obsData['X'], obsData['A'])
    X = X.loc[:, ['int', 'X', 'A', 'AX']]
    
    # Y contains the outcomes
    Y = obsData['Y']
    
    # Fit the regression model
    model = LinearRegression().fit(X, Y)
    
    # Calculate Qhat(H, 1)
    X_1 = X.copy(deep = True)
    X_1['A'] = 1
    Qhat1 = model.predict(X_1)
    
    # Calculate Qhat(H, 0)
    X_0 = X.copy(deep = True)
    X_0['A'] = 0
    Qhat0 = model.predict(X_0)
    
    # Calculate the treatment recommendation under the policy indexed by beta
    A_d = np.logical_or(obsData['X'] < beta[0], obsData['X'] > beta[1])*1.0
    
    # Calculate the pseudo value
    obsData['yhat'] = np.where(A_d == 1, Qhat1, 0) + np.where(A_d == 0, Qhat0, 0)
    
    # Estimate the value
#     vhat = (1/n)*sum(obsData['summand'] - obsData['weight']*obsData['yhat'])
    vhat = (1/n)*sum(obsData['yhat']) + (1/n)*sum((obsData['C_d']/obsData['pi_d'])*(obsData['Y'] - obsData['yhat']))
    
    return -1*vhat
    
# Simulation code --------------------------------------------------------------
w = [0.75, 1, 1.25][int(sys.argv[3])]
# Read in the true values for the evaluation of the GP
trueValues = pd.read_csv('../2_pipeline/3_simulation1D3_'+str(w)+'_trueValues.csv')
nObs = int(sys.argv[2])
evaluationEstimator = sys.argv[1]
L = 1000
outFileName = '3_simulation1D3_'+str(evaluationEstimator)+'_'+str(nObs)+'_'+str(w)+'.csv'

# Places to hold the things we want to keep
optDTR_param_holder = []
optDTR_value_holder = []
L_holder = []
norm_sup_holder = []
norm_1_holder = []
norm_2_holder = []

# Set the seed -------------------------------------
np.random.seed(1234)

for l in range(L):
    
    # Generate simulation data set ---------------------
    # Parameters for data generation
    gamma = [-0.5, 1]
    obsData = generateObservedData(nObs, w, gamma)
    obsData = pd.DataFrame(obsData).transpose()
 
    # Tidy up the dataframe with the "observed data"
    obsData = obsData.rename(columns = {0:'intercept', 1:'X', 2:'A', 3:'Y'})

    # Bayesian optimization ------------------------------
    noise = 0.01
    if evaluationEstimator == 'IPW':
        def computeIPW_internal(beta, obsData = obsData, nObs = nObs):
            return computeIPW(beta, obsData = obsData, n = nObs)
        ei_result = gp_minimize(computeIPW_internal,
                   [(0.0, 1.0), (0.0, 1.0)],
                   acq_func = "EI",
                   n_calls = 50,
                   n_random_starts = 50,
                   noise = noise)
    if evaluationEstimator == "sIPW":
        def computeStabilizedIPW_internal(beta, obsData = obsData, nObs = nObs):
            return computeStabilizedIPW(beta, obsData = obsData, n = nObs)
        ei_result = gp_minimize(computeStabilizedIPW_internal,
                   [(0.0, 1.0), (.0, 1.0)],
                   acq_func = "EI",
                   n_calls = 50,
                   n_random_starts = 50,
                   noise = noise)
    if evaluationEstimator == "gcomp":
        def computeRegEst_internal(beta, obsData = obsData, nObs = nObs):
            return computeRegEst(beta, obsData = obsData, n = nObs)
        ei_result = gp_minimize(computeRegEst_internal,
                   [(0.0, 1.0), (0.0, 1.0)],
                   acq_func = "EI",
                   n_calls = 50,
                   n_random_starts = 50,
                   noise = noise)
    if evaluationEstimator == "AIPWE":
        def computeAIPWE_internal(beta, obsData = obsData, nObs = nObs):
            return computeAIPWE(beta, obsData = obsData, n = nObs)
        ei_result = gp_minimize(computeAIPWE_internal,
                   [(0.0, 1.0), (0.0, 1.0)],
                   acq_func = "EI",
                   n_calls = 50,
                   n_random_starts = 50,
                   noise = noise)
    # Extract the relevant information
    optDTR_param = ei_result['x']
    optDTR_value = ei_result['fun']
    evaluation_X = ei_result['x_iters']
    evaluation_Y = ei_result['func_vals']

    # Fit a GP to the evaluation points --------------------
    kernel = 1.0 * Matern(length_scale = [1.0, 1.0], nu = 1.0) \
        + WhiteKernel(noise_level = 10, noise_level_bounds = (1e-10, 1e2))
    gpr = GaussianProcessRegressor(kernel = kernel, alpha = 0.0)
    gpr.fit(evaluation_X, evaluation_Y)

    # Get predictions across a fine grid of the parameter space
    diffs = []
    for i in range(trueValues.shape[0]):
        pred = gpr.predict(np.array(trueValues.loc[i, ['beta0', 'beta1']]).reshape(1, -1))
        diffs.append(-1*pred - trueValues.loc[i, 'value'])
    # Compute the distance between the prediction and the Truth (actual truth, not the evaluation truth)    
    norm_sup = max(np.abs(diffs))
    norm_1 = np.mean(np.abs(diffs))
    norm_2 = np.sqrt(sum(np.abs(diffs)**2)*(1/trueValues.shape[0]))

    # Update the holder lists
    optDTR_param_holder.append(optDTR_param)
    optDTR_value_holder.append(optDTR_value)
    L_holder.append(l)
    norm_sup_holder.append(norm_sup)
    norm_1_holder.append(norm_1)
    norm_2_holder.append(norm_2)

# Convert the list of arrays to a list
norm_sup_holder = [i[0] for i in norm_sup_holder]
norm_2_holder = [i[0] for i in norm_2_holder]

# Put the lists into a data frame
out1 = pd.DataFrame({'l':L_holder, 'optDTR_value':optDTR_value_holder, 'norm_sup':norm_sup_holder,
                     'norm_1':norm_1_holder, 'norm_2':norm_2_holder})
out2 = pd.DataFrame(optDTR_param_holder, columns = ['beta0', 'beta1'])
out = pd.concat([out1, out2], axis = 1)

# Write the dataframe to a csv
out.to_csv('../2_pipeline/'+outFileName)      
