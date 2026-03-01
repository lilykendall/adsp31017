"""
Name: LogisticRegression.py
Course: ADSP31017: Machine Learning I
Author: Ming-Long Lam, Ph.D.
Organization: University of Chicago
Last Modified: February 11, 2025
(C) All Rights Reserved.
"""

import numpy
import pandas
import statsmodels.api as smodel

from scipy.special import gammaln
from scipy.stats import norm, t

def SWEEPOperator (pDim, inputM, origDiag, sweepCol = None, tol = 1e-7):
    ''' Implement the SWEEP operator

    Parameter
    ---------
    pDim: dimension of matrix inputM, integer greater than one
    inputM: a square and symmetric matrix, numpy array
    origDiag: the original diagonal elements before any SWEEPing
    sweepCol: a list of columns numbers to SWEEP
    tol: singularity tolerance, positive real

    Return
    ------
    A: negative of a generalized inverse of input matrix
    aliasParam: a list of aliased rows/columns in input matrix
    nonAliasParam: a list of non-aliased rows/columns in input matrix
    '''

    if (sweepCol is None):
        sweepCol = range(pDim)

    aliasParam = []
    nonAliasParam = []

    A = numpy.copy(inputM)
    ANext = numpy.zeros((pDim,pDim))

    for k in sweepCol:
        Akk = A[k,k]
        pivot = tol * abs(origDiag[k])
        if (not numpy.isinf(Akk) and abs(Akk) >= pivot and pivot > 0.0):
            nonAliasParam.append(k)
            ANext = A - numpy.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / abs(Akk)
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
        A = ANext
    return (A, aliasParam, nonAliasParam)

def CramerV (xCat, yCat):
   ''' Calculate Cramer V statistic

   Argument:
   ---------
   xCat : a Pandas Series
   yCat : a Pandas Series

   Output:
   -------
   cramerV : Cramer V statistic
   '''

   obsCount = pandas.crosstab(index = xCat, columns = yCat, margins = False, dropna = True)
   xNCat = obsCount.shape[0]
   yNCat = obsCount.shape[1]
    
   if (xNCat > 1 and yNCat > 1):
      cTotal = obsCount.sum(axis = 1)
      rTotal = obsCount.sum(axis = 0)
      nTotal = numpy.sum(rTotal)
      expCount = numpy.outer(cTotal, (rTotal / nTotal))

      # Calculate the Chi-Square statistics
      chiSqStat = ((obsCount - expCount)**2 / expCount).to_numpy().sum()
      cramerV = chiSqStat / nTotal / (min(xNCat, yNCat) - 1.0)
      cramerV = numpy.sqrt(cramerV)
   else:
      cramerV = numpy.NaN

   return (cramerV)

def create_interaction (df1, df2):
    ''' Return the columnwise product of two dataframes (must have same number of rows)

    Parameter
    ---------
    df1: first input data frame
    df2: second input data frame

    Return
    ------
    outDF: the columnwise product of two dataframes
    '''

    name1 = df1.columns
    name2 = df2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        outName = col1 + ' * ' + name2
        outDF[outName] = df2.multiply(df1[col1], axis = 'index')
    return(outDF)

def paste_interaction (effectName):
    name1 = ''
    name2 = ''
    aName = effectName.strip() 
    ipos = aName.find('*')
    if (ipos >= 0):
        name1 = aName[:ipos].strip()
        name2 = aName[(ipos+1):].strip()
        if (len(name1) == 0 and len(name2) > 0):
            name1 = name2
            name2 = ''
    else:
        name1 = aName

    return ([name1, name2])

def construct_X_from_spec (data, catPred, intPred, modelSpec, qIntercept = True):

    # Generate the model matrix in the order of model specification
    modelX = data[[]]
    for effect in modelSpec:
        name_list =  paste_interaction(effect)
        name1 = name_list[0]
        if (len(name1) > 0):
            if (name1 in catPred):
               df1 = pandas.get_dummies(data[[name1]].astype('category'), dtype = float)
            else:
               df1 = data[[name1]]
        else:
            df1 = None

        name2 = name_list[1]
        if (len(name2) > 0):
            if (name2 in catPred):
               df2 = pandas.get_dummies(data[[name2]].astype('category'), dtype = float)
            else:
               df2 = data[[name2]]
        else:
           df2 = None

        if (df1 is not None and df2 is not None):
           modelX = modelX.join(create_interaction (df1, df2))
        elif (df1 is not None):
           modelX = modelX.join(df1)

    if (qIntercept):
        modelX.insert(0, 'Intercept', 1.0)

    return (modelX)

def BinaryLogisticRegression (trainData, catPred, intPred, binaryLabel, eventCategory, \
                              modelSpec, qIntercept = True, \
                              maxIter = 500, maxStep = 7, tolLLK = 1e-3, tolEpsilon = 1e-10, tolSweep = 1e-7):
    ''' Train a Binary Logistic Regression model

    Parameters
    ----------
    trainData: A Pandas DataFrame, rows are observations, columns are features
    catPred: A list of names to columns of trainData, designated as categorical predictors
    intPred: A list of names to columns of trainData, designated as interval predictors
    binaryLabel: A name to a column of trainData, desigated as the binary target variable
    eventCategory: A string that contains the event category
    modelSpec: A list of model effect specifications (main effect and two-way interactions only)
    qIntercept: If True, the model will include the Intercept term. Otherwise, the model has no Intercept term.
    maxIter: Maximum number of iterations
    maxStep: Maximum number of step-halving
    tolLLK: Minimum absolute difference to get a successful step-halving
    tolEpsilon: A surrogate value for zero
    tolSweep: Tolerance for SWEEP Operator

    Return List
    -----------
    outCoefficient: a Pandas DataFrame of regression coefficients, standard errors, and confidence interval
    outCovb: a Pandas DataFrame of covariance matrix of regression coefficients
    outCorb: a Pandas DataFrame of correlation matrix of regression coefficients
    llk: log-likelihood value
    nonAliasParam: a list of non-aliased rows/columns in input matrix
    outIterationTable: a Pandas DataFrame of iteration history table
    labelCategories: A Pandas Series of label variable's categories
    predprob_df: a Pandas DataFrame of event count, total count, and the predicted probabilities
    measure_assoc: a list with 0: McFadden R-squares, 1: Cox & Snell R-squares,
                   2: Nagelkerke R-squares, and  3: Tjur Coefficient of Discrimination
    '''

    predictor_name = catPred + intPred

    # Generate the crosstabulation of predictors by binary label
    if (len(predictor_name) > 0):
        xtab = pandas.crosstab(index = [trainData[pred] for pred in predictor_name],
                               columns = trainData[binaryLabel]).reset_index(drop = False)
        predictor_df = xtab[predictor_name]
        label_count = xtab.drop(columns = predictor_name)
        n_count = label_count.sum(axis = 1)
    else:
        predictor_df = pandas.DataFrame()
        label_count = trainData[binaryLabel].value_counts()
        n_count = label_count.sum()

    # Extract columns of he crosstabulation
    event_count = label_count[eventCategory]
    nonevent_count = n_count -  event_count

    # Generate the model matrix in the order of model specification
    modelX = construct_X_from_spec (predictor_df, catPred, intPred, modelSpec, qIntercept)

    # Initialize the predicted event and non-event probabilities
    if (qIntercept):
        obs_event_prob = numpy.sum(event_count) / numpy.sum(n_count)
        obs_nonevent_prob = 1.0 - obs_event_prob
    else:
        obs_event_prob = 0.5
        obs_nonevent_prob = 0.5

    n_value_comb = modelX.shape[0]
    n_param = modelX.shape[1]
    param_name = modelX.columns

    modelXT = modelX.transpose()

    # Initialize predicted probabilities, parameter estimates, and log-likelihood value
    event_prob = numpy.full(n_value_comb, obs_event_prob)
    nonevent_prob = numpy.full(n_value_comb, obs_nonevent_prob)

    beta = pandas.Series(numpy.zeros(n_param), index = param_name)
    beta[param_name[0]] = numpy.log(obs_event_prob / obs_nonevent_prob)

    llk_constant = numpy.sum(gammaln(n_count + 1.0) - gammaln(event_count + 1.0) - gammaln(nonevent_count + 1.0))
    llk_kernel = numpy.sum(event_count * numpy.log(event_prob) + nonevent_count * numpy.log(nonevent_prob))
    llk = llk_kernel + llk_constant
    llk_kernel_0 = llk_kernel

    # Prepare the iteration history table (Iteration #, Log-Likelihood, N Step-Halving, Beta)
    itList = [0, llk, 0]
    itList.extend(beta)
    iterTable = [itList]

    for it in range(maxIter):
        expected_event_count = n_count * event_prob
        gradient = modelXT.dot(event_count - expected_event_count)
        dispersion = expected_event_count * nonevent_prob
        hessian = - modelXT.dot(dispersion.values.reshape((n_value_comb,1)) * modelX)
        orig_diag = numpy.diag(hessian)
        invhessian, aliasParam, nonAliasParam = SWEEPOperator (n_param, hessian, orig_diag, sweepCol = range(n_param), tol = tolSweep)
        invhessian[:, aliasParam] = 0.0
        invhessian[aliasParam, :] = 0.0
        delta = numpy.matmul(-invhessian, gradient)
        step = 1.0
        for iStep in range(maxStep):
            beta_next = beta - step * delta
            nu_next = modelX.dot(beta_next)
            exp_nu_next = numpy.exp(nu_next)

            event_prob_next = 1.0 / (1.0 + (1.0 / exp_nu_next))
            nonevent_prob_next = 1.0 / (1.0 + exp_nu_next)
            llk_next = numpy.sum(event_count * numpy.log(event_prob_next) + nonevent_count * numpy.log(nonevent_prob_next)) + llk_constant
            if ((llk_next - llk) > - tolLLK):
                break
            else:
                step = 0.5 * step

        diffBeta = beta_next - beta
        beta = beta_next
        llk = llk_next
        event_prob = event_prob_next
        nonevent_prob = nonevent_prob_next
        itList = [it+1, llk, iStep]
        itList.extend(beta)
        iterTable.append(itList)
        if (numpy.linalg.norm(diffBeta) < tolEpsilon or abs(llk) < tolEpsilon):
            break

    it_name = ['Iteration', 'Log-Likelihood', 'N Step-Halving']
    it_name.extend(param_name)
    outIterationTable = pandas.DataFrame(iterTable, columns = it_name)

    # Final covariance matrix
    stderr = numpy.sqrt(numpy.diag(invhessian))
    z95 = norm.ppf(0.975)

    # Final parameter estimates
    outCoefficient = pandas.DataFrame(beta, index = param_name, columns = ['Estimate'])
    outCoefficient['Standard Error'] = stderr
    outCoefficient['Lower 95% CI'] = beta - z95 * stderr
    outCoefficient['Upper 95% CI'] = beta + z95 * stderr

    outCovb = pandas.DataFrame(invhessian, index = param_name, columns = param_name)

    temp_m1_ = numpy.outer(stderr, stderr)
    outCorb = pandas.DataFrame(numpy.divide(invhessian, temp_m1_, out = numpy.zeros_like(invhessian), where = (temp_m1_ != 0.0)),
                               index = param_name, columns = param_name)
    
    labelCategories = label_count.columns

    predprob_df = predictor_df.join(label_count)
    predprob_df = predprob_df.join(pandas.DataFrame({'Total': n_count, 'Predicted Event Probability': event_prob, \
                                                     'Predicted Non-Event Probability': nonevent_prob}))

    # Calculate the measure of association
    llk_kernel = llk - llk_constant
    n_sample = trainData.shape[0]

    R_MF = 1.0 - (llk_kernel / llk_kernel_0)

    R_CS = (2.0 / n_sample) * (llk_kernel_0 - llk_kernel)
    R_CS = 1.0 - numpy.exp(R_CS)

    upbound = (2.0 / n_sample) * llk_kernel_0
    upbound = 1.0 - numpy.exp(upbound)
    R_N = R_CS / upbound

    event_count = label_count[eventCategory]
    S1 = numpy.sum(event_prob * event_count) / numpy.sum(event_count)
    
    nonevent_count = n_count - event_count
    S0 = numpy.sum(event_prob * nonevent_count) / numpy.sum(nonevent_count)
    R_TJ = S1 - S0

    measure_assoc = [R_MF, R_CS, R_N, R_TJ]

    return ([outCoefficient, outCovb, outCorb, llk, nonAliasParam, outIterationTable, labelCategories, predprob_df, measure_assoc])

def MultinominalLogisticRegression (trainData, catPred, intPred, nominalLabel, modelSpec, qIntercept = True, \
                                    maxIter = 10000, tolEpsilon = 1e-10, tolSweep = 1e-7):
    ''' Train a Multinominal Logistic Regression model

    Parameters
    ----------
    trainData: A Pandas DataFrame, rows are observations, columns are features
    catPred: A list of names to columns of trainData, designated as categorical predictors
    intPred: A list of names to columns of trainData, designated as interval predictors
    nominalLabel: A name to a column of trainData, desigated as the nominal target variable
    modelSpec: A list of model effect specifications (main effect and two-way interactions only)
    qIntercept: If True, the model will include the Intercept term. Otherwise, the model has no Intercept term.
    maxIter: Maximum number of iterations
    tolEpsilon: A surrogate value for zero
    tolSweep: Tolerance for SWEEP Operator

    Return List
    ----------
    model_fit: Python object of the fitted model
    model_LLK: Model final log-likelihood value
    model_DF: Model final degree of freedom
    model_parameter: Pandas dataframe of parameter estimates (one column per label category)
    aliasParam: a list of positions of aliased parameters in model matrix
    nonAliasParam: a list of positions of non-aliased parameters in model matrix
    labelCategories: a list of label categories
    '''

    predictor_name = catPred + intPred
    assert len(predictor_name) > 0, "No predictors are specified."

    # Create the label categories in descending order of frequency
    label_count = trainData[nominalLabel].value_counts()
    labelCategories = label_count.index.to_list()
    n_label_categories = len(labelCategories)
    y = trainData[nominalLabel].map({value: index for index, value in enumerate(labelCategories)}).astype('category')

    # Generate the model matrix in the order of model specification
    X_full = construct_X_from_spec (trainData[predictor_name], catPred, intPred, modelSpec, qIntercept)
    n_value_comb = X_full.shape[0]
    n_param = X_full.shape[1]
    param_name = X_full.columns

    XtX = X_full.transpose().dot(X_full)
    orig_diag = numpy.diag(XtX)
    XtXGinv, aliasParam, nonAliasParam = SWEEPOperator (n_param, XtX, orig_diag, sweepCol = range(n_param), tol = tolSweep)

    # Train a multinominal logistic model
    X_reduce = X_full.iloc[:, list(nonAliasParam)]
    model_obj = smodel.MNLogit(y, X_reduce)
    model_fit = model_obj.fit(method = 'newton', maxiter = maxIter, tol = tolEpsilon, full_output = True, disp = True)
    model_LLK = model_fit.llf
    model_DF = len(nonAliasParam) * (model_fit.J - 1)

    # MNLogit uses the FIRST label category as reference
    model_parameter = model_fit.params.rename(mapper = {index: value for index, value in enumerate(labelCategories[1:])}, axis = 1)
    model_parameter = pandas.merge(pandas.DataFrame(index = param_name), model_parameter, left_index=True, right_index=True, how='outer').fillna(0.0)

    # Return model statistics
    return ([model_fit, model_LLK, model_DF, model_parameter, aliasParam, nonAliasParam, labelCategories])
