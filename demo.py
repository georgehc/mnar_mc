"""
Benchmarking matrix completion algorithms and their debiased variants
(including ones following our approach of debiasing matrix completion with
propensity scores estimated via matrix completion)

Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (weima@cmu.edu)
"""
import copy
import json
import multiprocessing
import numpy as np
import os
import random
import sys
import warnings
os.environ['QT_QPA_PLATFORM']='offscreen'
warnings.filterwarnings("ignore")

from subprocess import DEVNULL, call
from surprise import Dataset, Reader, SVD, SVDpp, accuracy, KNNBasic
from surprise.model_selection import GridSearchCV, train_test_split
from models import WeightedSoftImputeWrapper, WeightedSoftImputeALSWrapper, \
        ExpoMFWrapper, DoublyWeightedTraceNormWrapper, WeightedMaxNormWrapper
from weighted_surprise_prediction_algorithms import WeightedSVD, WeightedSVDpp


# prevent numpy/scipy/etc from only using a single processor; see:
# https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
# (note that this is unix/linux only and should silently error on other
# platforms)
call(['taskset', '-p', '0x%s' % ('f' * int(np.ceil(os.cpu_count() / 4))),
      '%d' % os.getpid()], stdout=DEVNULL, stderr=DEVNULL)


output_dir = 'output'
algs_to_run = ['1bitMC-PMF']
# algs_to_run = ['PMF', 'SVD', 'SVDpp', 'SoftImpute', 'MaxNorm', 'WTN', 'ExpoMF',
#                '1bitMC-PMF', '1bitMC-SVD', '1bitMC-SVDpp', '1bitMC-SoftImpute',
#                '1bitMC-MaxNorm', '1bitMC-WTN',
#                'NB-PMF', 'NB-SVD', 'NB-SVDpp', 'NB-SoftImpute',
#                'NB-MaxNorm', 'NB-WTN',
#                'LR-PMF', 'LR-SVD', 'LR-SVDpp', 'LR-SoftImpute',
#                'LR-MaxNorm', 'LR-WTN']
num_jobs_for_cross_val = -1  # -1 means use all cores

if len(sys.argv) != 2:
    print('Usage: python %s <dataset name>' % sys.argv[0])
    sys.exit()

dataset_name = sys.argv[1]

os.makedirs(output_dir, exist_ok=True)

if dataset_name.startswith('ml'):
    algs_to_run = [algo_name for algo_name in algs_to_run
                   if not (algo_name.startswith('NB') \
                           or algo_name.startswith('LR'))]
elif dataset_name.startswith('steck'):
    algs_to_run = [algo_name for algo_name in algs_to_run
                   if not algo_name.startswith('LR')]
print("Algorithms to run:", algs_to_run)
print()

# hyperparameter search grids
param_grids = {
        'SVD':
            {'reg_all': [.002, .02, .2],
             'n_epochs': [20, 200],
             'random_state': [0]},
        'PMF':
            {'reg_all': [.002, .02, .2],
             'n_epochs': [20, 200],
             'biased': [False],
             'random_state': [0]},
        'SVDpp':
            {'reg_all': [.002, .02, .2],
             'n_epochs': [20, 200],
             'random_state': [0]},
        'SoftImpute': {'max_rank': [40],
                       'lmbda': [1, 10, 100],
                       'max_iter': [100],
                       'min_value': [1],
                       'max_value': [5]},
        'SoftImputeALS': {'max_rank': [40],
                          'lmbda': [1, 10, 100],
                          'max_iter': [100],
                          'verbose': [False]},
        'MaxNorm' : {'n_components': [10, 20, 40],
                     'R': [1, 10, 100],
                     'alpha': [5, 10, 20],
                     'max_iter': [100],
                     'random_state': [0]},
        'WTN'     : {'n_components': [10, 20, 40],
                     'lmbda' : [1, 10, 100],
                     'max_iter': [100],
                     'random_state': [0]},
        'ExpoMF'  : {'n_components': [10, 20, 40],
                     'random_state': [0]},
        'KNN'     : {'k' : [2, 5, 10, 20, 40],
                     'verbose': [False]},
    }
one_bit_mc_grid = {
        'propensity_scores': ['1bitmc'],
        'one_bit_mc_max_rank': [40],
        'one_bit_mc_tau': [1., 10., 100., 1000.],
        'one_bit_mc_gamma': [3],
    }
one_bit_mc_mod_grid = {
        'propensity_scores': ['1bitmc_mod'],
        'one_bit_mc_max_rank': [40],
        'one_bit_mc_tau': [1., 10., 100., 1000.],
        'one_bit_mc_gamma': [3],
    }

param_grids['1bitMC-SVD'] = {**param_grids['SVD'], **one_bit_mc_grid}
param_grids['1bitMC-PMF'] = {**param_grids['PMF'], **one_bit_mc_grid}
param_grids['1bitMC-SVDpp'] = {**param_grids['SVDpp'], **one_bit_mc_grid}
param_grids['1bitMC-SoftImpute'] = {**param_grids['SoftImpute'],
                                    **one_bit_mc_grid}
param_grids['1bitMC-SoftImputeALS'] = {**param_grids['SoftImputeALS'],
                                       **one_bit_mc_grid}
param_grids['1bitMC-MaxNorm'] = {**param_grids['MaxNorm'], **one_bit_mc_grid}
param_grids['1bitMC-WTN'] = {**param_grids['WTN'], **one_bit_mc_grid}

param_grids['1bitMCmod-SVD'] = {**param_grids['SVD'], **one_bit_mc_mod_grid}
param_grids['1bitMCmod-PMF'] = {**param_grids['PMF'], **one_bit_mc_mod_grid}
param_grids['1bitMCmod-SVDpp'] = {**param_grids['SVDpp'],
                                  **one_bit_mc_mod_grid}
param_grids['1bitMCmod-SoftImpute'] = {**param_grids['SoftImpute'],
                                       **one_bit_mc_mod_grid}
param_grids['1bitMCmod-SoftImputeALS'] = {**param_grids['SoftImputeALS'],
                                          **one_bit_mc_mod_grid}
param_grids['1bitMCmod-MaxNorm'] = {**param_grids['MaxNorm'],
                                    **one_bit_mc_mod_grid}
param_grids['1bitMCmod-WTN'] = {**param_grids['WTN'],
                                **one_bit_mc_mod_grid}

param_grids['NB-SVD'] = {**param_grids['SVD']}
param_grids['NB-PMF'] = {**param_grids['PMF']}
param_grids['NB-SVDpp'] = {**param_grids['SVDpp']}
param_grids['NB-SoftImpute'] = {**param_grids['SoftImpute']}
param_grids['NB-SoftImputeALS'] = {**param_grids['SoftImputeALS']}
param_grids['NB-MaxNorm'] = {**param_grids['MaxNorm']}
param_grids['NB-WTN'] = {**param_grids['WTN']}

param_grids['LR-SVD'] = {**param_grids['SVD']}
param_grids['LR-PMF'] = {**param_grids['PMF']}
param_grids['LR-SVDpp'] = {**param_grids['SVDpp']}
param_grids['LR-SoftImpute'] = {**param_grids['SoftImpute']}
param_grids['LR-SoftImputeALS'] = {**param_grids['SoftImputeALS']}
param_grids['LR-MaxNorm'] = {**param_grids['MaxNorm']}
param_grids['LR-WTN'] = {**param_grids['WTN']}

# note that the probabilistic matrix factorization (PMF) algorithm by
# Salakhutdinov and Minh (2007) also uses Surprise's "SVD" class (but has the
# parameter `biased` set to False; this is shown in `param_grids[PMF]`)
algorithms = {
        'PMF': SVD,
        'SVD': SVD,
        'SVDpp': SVDpp,
        'SoftImpute': WeightedSoftImputeWrapper,
        'SoftImputeALS': WeightedSoftImputeALSWrapper,
        'MaxNorm': WeightedMaxNormWrapper,
        'WTN': DoublyWeightedTraceNormWrapper,
        'ExpoMF': ExpoMFWrapper,
        'KNN': KNNBasic,
        '1bitMC-PMF': WeightedSVD,
        '1bitMC-SVD': WeightedSVD,
        '1bitMC-SVDpp': WeightedSVDpp,
        '1bitMC-SoftImpute': WeightedSoftImputeWrapper,
        '1bitMC-SoftImputeALS': WeightedSoftImputeALSWrapper,
        '1bitMC-MaxNorm': WeightedMaxNormWrapper,
        '1bitMC-WTN': DoublyWeightedTraceNormWrapper,
        '1bitMCmod-PMF': WeightedSVD,
        '1bitMCmod-SVD': WeightedSVD,
        '1bitMCmod-SVDpp': WeightedSVDpp,
        '1bitMCmod-SoftImpute': WeightedSoftImputeWrapper,
        '1bitMCmod-SoftImputeALS': WeightedSoftImputeALSWrapper,
        '1bitMCmod-MaxNorm': WeightedMaxNormWrapper,
        '1bitMCmod-WTN': DoublyWeightedTraceNormWrapper,
        'NB-PMF': WeightedSVD,
        'NB-SVD': WeightedSVD,
        'NB-SVDpp': WeightedSVDpp,
        'NB-SoftImpute': WeightedSoftImputeWrapper,
        'NB-SoftImputeALS': WeightedSoftImputeALSWrapper,
        'NB-MaxNorm': WeightedMaxNormWrapper,
        'NB-WTN': DoublyWeightedTraceNormWrapper,
        'LR-PMF': WeightedSVD,
        'LR-SVD': WeightedSVD,
        'LR-SVDpp': WeightedSVDpp,
        'LR-SoftImpute': WeightedSoftImputeWrapper,
        'LR-SoftImputeALS': WeightedSoftImputeALSWrapper,
        'LR-MaxNorm': WeightedMaxNormWrapper,
        'LR-WTN': DoublyWeightedTraceNormWrapper,
    }

# some deterministic seed setting to get reproducible results
import hashlib
random.seed(0)
np.random.seed(0)
# create a specific random seed per algorithm by hashing the algorithm name
algorithm_deterministic_seeds = \
    {algo_name: abs(hash(algo_name)) % (10 ** 8)
     for algo_name in algs_to_run}


# -----------------------------------------------------------------------------
# Construct training data
#

def construct_inner_P(p_data, train_data):
    train_set = train_data.build_full_trainset()
    p_set = p_data.build_full_trainset()
    assert (train_set.n_users == p_set.n_users)
    assert (train_set.n_items == p_set.n_items)
    P = np.zeros((train_set.n_users, train_set.n_items))
    for u, i, p in p_set.all_ratings():
        P[train_set.to_inner_uid(p_set.to_raw_uid(u)),
          train_set.to_inner_iid(p_set.to_raw_iid(i))] = p
    return P

if dataset_name.startswith('ml-100k'):
    num = np.int(dataset_name.split('-')[-1])
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file(os.path.join('ml-100k',
                                               str(num), 'observed_X.txt'),
                                  reader=reader)

elif dataset_name == 'coat':
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file('coat/train_surprise_format.csv',
                                  reader=reader)
    reader = Reader(line_format='user item rating', sep='\t')
    NB_p_data = Dataset.load_from_file(os.path.join('coat/P_NB.txt'),
                                       reader=reader)
    NB_P = construct_inner_P(NB_p_data, data)
    reader = Reader(line_format='user item rating', sep='\t')
    LR_p_data = Dataset.load_from_file(os.path.join('coat/P_LR.txt'),
                                       reader=reader)
    LR_P = construct_inner_P(LR_p_data, data)

    for algo_name in param_grids:
        if algo_name.startswith('NB-'):
            param_grids[algo_name]['propensity_scores'] = [NB_P]

    for algo_name in param_grids:
        if algo_name.startswith('LR-'):
            param_grids[algo_name]['propensity_scores'] = [LR_P]

elif dataset_name.startswith('useritemfeature'):
    num = np.int(dataset_name.split('-')[1])
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file(os.path.join('useritemfeature',
                                               str(num), 'observed_X.txt'),
                                  reader=reader)
    reader = Reader(line_format='user item rating', sep='\t')
    NB_p_data = Dataset.load_from_file(os.path.join('useritemfeature',
                                                    str(num), 'P_NB.txt'),
                                       reader=reader)
    NB_P = construct_inner_P(NB_p_data, data)
    reader = Reader(line_format='user item rating', sep='\t')
    LR_p_data = Dataset.load_from_file(os.path.join('useritemfeature',
                                                    str(num), 'P_LR.txt'),
                                       reader=reader)
    LR_P = construct_inner_P(LR_p_data, data)

    for algo_name in param_grids:
        if algo_name.startswith('NB-'):
            param_grids[algo_name]['propensity_scores'] = [NB_P]

    for algo_name in param_grids:
        if algo_name.startswith('LR-'):
            param_grids[algo_name]['propensity_scores'] = [LR_P]

elif dataset_name.startswith('steck'):
    num = np.int(dataset_name.split('-')[1])
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file(os.path.join('steck', str(num),
                                               'observed_X.txt'),
                                  reader=reader)
    reader = Reader(line_format='user item rating', sep='\t')
    NB_p_data = Dataset.load_from_file(os.path.join('steck',
                                                    str(num), 'P_NB.txt'),
                                       reader=reader)
    NB_P = construct_inner_P(NB_p_data, data)

    for algo_name in param_grids:
        if algo_name.startswith('NB-'):
            param_grids[algo_name]['propensity_scores'] = [NB_P]

else:
    raise Exception('Dataset not supported: ' + dataset_name)


# -----------------------------------------------------------------------------
# Hyperparameter search via cross-validation on training data
#

best_algs_rmse = {}
best_algs_mae = {}
grid_searches = {}

for algo_name in algs_to_run:
    print('[Dataset: %s - algorithm: %s - grid search CV]'
          % (dataset_name, algo_name), flush=True)
    random.seed(algorithm_deterministic_seeds[algo_name])
    np.random.seed(algorithm_deterministic_seeds[algo_name])
    grid_search = GridSearchCV(algorithms[algo_name], param_grids[algo_name],
                               measures=['rmse', 'mae'], cv=5,
                               n_jobs=num_jobs_for_cross_val)
    grid_search.fit(data)

    grid_searches[algo_name] = grid_search
    best_algs_rmse[algo_name] = grid_search.best_estimator['rmse']
    best_algs_mae[algo_name] = grid_search.best_estimator['mae']

    print('Best parameters found:', flush=True)
    best_params = copy.deepcopy(grid_searches[algo_name].best_params)
    if algo_name.startswith('NB-') or algo_name.startswith('LR-'):
        if 'propensity_scores' in best_params['rmse']:
            del best_params['rmse']['propensity_scores']
        if 'propensity_scores' in best_params['mae']:
            del best_params['mae']['propensity_scores']
    print(json.dumps(best_params, indent=4, sort_keys=True), flush=True)
    print()


# -----------------------------------------------------------------------------
# Now train on full dataset per algorithm (using best hyperparameters found via
# cross-validation) and evaluate error on test data
#

train_set = data.build_full_trainset()
if dataset_name.startswith('ml-100k'):
    num = np.int(dataset_name.split('-')[-1])
    reader = Reader(line_format='user item rating', sep='\t')
    test_data = Dataset.load_from_file(os.path.join('ml-100k',
                                                    str(num), 'test_X.txt'),
                                       reader=reader)
    test_set = test_data.construct_testset(test_data.raw_ratings)
elif dataset_name == 'coat':
    test_data = Dataset.load_from_file('coat/test_surprise_format.csv',
                                       reader=reader)
    test_set = test_data.construct_testset(test_data.raw_ratings)
elif dataset_name.startswith('useritemfeature'):
    num = np.int(dataset_name.split('-')[1])
    reader = Reader(line_format='user item rating', sep='\t')
    test_data = Dataset.load_from_file(os.path.join('useritemfeature',
                                                    str(num), 'whole_X.txt'),
                                       reader=reader)
    test_set = test_data.construct_testset(test_data.raw_ratings)
elif dataset_name.startswith('steck'):
    num = np.int(dataset_name.split('-')[1])
    reader = Reader(line_format='user item rating', sep='\t')
    test_data = Dataset.load_from_file(os.path.join('steck', str(num),
                                                    'whole_X.txt'),
                                       reader=reader)
    test_set = test_data.construct_testset(test_data.raw_ratings)


def surprise_predictions_to_matrix(predictions):
    predicted_ratings = []
    for prediction in predictions:
        try:
            u = train_set.to_inner_uid(prediction.uid)
            i = train_set.to_inner_iid(prediction.iid)
        except:
            continue
        r = prediction.est
        predicted_ratings.append((u, i, r))
    return np.array(predicted_ratings)

# RMSE/MSE
print('[Dataset: %s - test set RMSE/MSE]' % dataset_name, flush=True)
for algo_name in algs_to_run:
    random.seed(algorithm_deterministic_seeds[algo_name] + 1)
    np.random.seed(algorithm_deterministic_seeds[algo_name] + 1)
    algo = best_algs_rmse[algo_name]
    algo.fit(train_set)
    predictions = algo.test(test_set)
    np.savetxt(os.path.join(output_dir,
                            '%s_%s_mse_predictions.txt'
                            % (dataset_name, algo_name)),
               surprise_predictions_to_matrix(predictions))

    if algo_name.startswith('1bitMC'):
        propensity_estimates = algo.propensity_estimates
        np.savetxt(os.path.join(output_dir,
                                '%s_%s_mse_propensity_estimates.txt'
                                % (dataset_name, algo_name)),
                   propensity_estimates)

    print(algo_name,
          'RMSE:', accuracy.rmse(predictions, verbose=False),
          'MSE:', accuracy.mse(predictions, verbose=False),
          flush=True)

print()

# MAE
print('[Dataset: %s - test set MAE]' % dataset_name, flush=True)
for algo_name in algs_to_run:
    random.seed(algorithm_deterministic_seeds[algo_name] + 2)
    np.random.seed(algorithm_deterministic_seeds[algo_name] + 2)
    algo = best_algs_mae[algo_name]
    algo.fit(train_set)
    predictions = algo.test(test_set)
    np.savetxt(os.path.join(output_dir,
                            '%s_%s_mae_predictions.txt'
                            % (dataset_name, algo_name)),
               surprise_predictions_to_matrix(predictions))

    if algo_name.startswith('1bitMC'):
        propensity_estimates = algo.propensity_estimates
        np.savetxt(os.path.join(output_dir,
                                '%s_%s_mae_propensity_estimates.txt'
                                % (dataset_name, algo_name)),
                   propensity_estimates)

    print(algo_name,
          'MAE:', accuracy.mae(predictions, verbose=False),
          flush=True)

print()
