"""
Benchmarking matrix completion algorithms and their debiased variants
(including ones following our approach of debiasing matrix completion with
propensity scores estimated via matrix completion)

Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (wei.w.ma@polyu.edu.hk)
"""
import copy
import hashlib
import json
import numpy as np
import os
import random
import sys
import warnings
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
warnings.filterwarnings("ignore")

from subprocess import DEVNULL, call
from surprise import Dataset, Reader, SVD, SVDpp, accuracy, KNNBasic
from surprise.model_selection import GridSearchCV, train_test_split, KFold
from models import WeightedSoftImputeWrapper, WeightedSoftImputeALSWrapper, \
        ExpoMFWrapper, DoublyWeightedTraceNormWrapper, WeightedMaxNormWrapper
from weighted_surprise_prediction_algorithms import WeightedSVD, WeightedSVDpp


# prevent numpy/scipy/etc from only using a single processor; see:
# https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
# (note that this is unix/linux only and should silently error on other
# platforms)
try:
    call(['taskset', '-p', '0x%s' % ('f' * int(np.ceil(os.cpu_count() / 4))),
          '%d' % os.getpid()], stdout=DEVNULL, stderr=DEVNULL)
except Exception:
    pass


output_dir = 'output'
algs_to_run = ['PMF',
               '1bitMC-PMF',
               'rowcf-PMF',
               '1bitMC-rowcf-PMF',
               'SVD',
               '1bitMC-SVD',
               'rowcf-SVD',
               '1bitMC-rowcf-SVD',
               'SVDpp',
               '1bitMC-SVDpp',
               'rowcf-SVDpp',
               '1bitMC-rowcf-SVDpp',
               'SoftImpute',
               '1bitMC-SoftImpute',
               'rowcf-SoftImpute',
               '1bitMC-rowcf-SoftImpute',
               'WTN',
               '1bitMC-WTN',
               'rowcf-WTN',
               '1bitMC-rowcf-WTN',
               'MaxNorm',
               '1bitMC-MaxNorm',
               'rowcf-MaxNorm',
               '1bitMC-rowcf-MaxNorm',
               'ExpoMF']
num_jobs_for_cross_val = -1  # -1 means use all cores

if len(sys.argv) != 2:
    print('Usage: python %s <dataset name>' % sys.argv[0])
    sys.exit()

dataset_name = sys.argv[1]

os.makedirs(output_dir, exist_ok=True)

if dataset_name.startswith('ml') or dataset_name.startswith('binary'):
    algs_to_run = [algo_name for algo_name in algs_to_run
                   if not (algo_name.startswith('NB') \
                           or algo_name.startswith('LR'))]
elif dataset_name.startswith('steck') or dataset_name.startswith('rowcluster'):
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
                     'R': [2, 8, 32, 128],
                     'init_std': [0.01, 0.001, 0.0001],
                     'min_value': [1],
                     'max_value': [5],
                     'max_iter': [500],
                     'random_state': [0]},
        'WTN'     : {'n_components': [10, 20, 40],
                     'init_std': [0.01, 0.001, 0.0001],
                     'lmbda' : [1, 10, 100],
                     'max_iter': [500],
                     'min_value': [1],
                     'max_value': [5],
                     'random_state': [0]},
        'ExpoMF'  : {'n_components': [10, 20, 40],
                     'random_state': [0]},
        'KNN'     : {'k' : [2, 5, 10, 20, 40],
                     'verbose': [False]},
    }
log_reg_grid = {
        'min_prob': [.05]
    }
one_bit_mc_grid = {
        'propensity_scores': ['1bitmc'],
        'one_bit_mc_max_rank': [40],
        'one_bit_mc_tau': [10., 20.],
        'one_bit_mc_approx_lr': [0.1],  # only needed for approx
        'one_bit_mc_gamma': [3],
        'approx': [True]
    }
one_bit_mc_mod_grid = {
        'propensity_scores': ['1bitmc_mod'],
        'one_bit_mc_max_rank': [40],
        'one_bit_mc_tau': [10., 20.],
        'one_bit_mc_gamma': [3],
    }
rowcf_grid = {
        'propensity_scores': ['rowcf'],
        'row_cf_k' : [1, 16, 64],
        'row_cf_thres' : [0.01, 0.1],
        'approx': [True]
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

param_grids['LR-SVD'] = {**param_grids['SVD'], **log_reg_grid}
param_grids['LR-PMF'] = {**param_grids['PMF'], **log_reg_grid}
param_grids['LR-SVDpp'] = {**param_grids['SVDpp'], **log_reg_grid}
param_grids['LR-SoftImpute'] = {**param_grids['SoftImpute'], **log_reg_grid}
param_grids['LR-SoftImputeALS'] = {**param_grids['SoftImputeALS'],
                                   **log_reg_grid}
param_grids['LR-MaxNorm'] = {**param_grids['MaxNorm'], **log_reg_grid}
param_grids['LR-WTN'] = {**param_grids['WTN'], **log_reg_grid}

param_grids['rowcf-SVD'] = {**param_grids['SVD'], **rowcf_grid}
param_grids['rowcf-PMF'] = {**param_grids['PMF'], **rowcf_grid}
param_grids['rowcf-SVDpp'] = {**param_grids['SVDpp'], **rowcf_grid}
param_grids['rowcf-SoftImpute'] = {**param_grids['SoftImpute'],
                                   **rowcf_grid}
param_grids['rowcf-SoftImputeALS'] = {**param_grids['SoftImputeALS'],
                                      **rowcf_grid}
param_grids['rowcf-MaxNorm'] = {**param_grids['MaxNorm'], **rowcf_grid}
param_grids['rowcf-WTN'] = {**param_grids['WTN'], **rowcf_grid}

param_grids['1bitMC-rowcf-SVD'] = {**param_grids['SVD'], **rowcf_grid, **one_bit_mc_grid}
param_grids['1bitMC-rowcf-SVD']['propensity_scores'] = ['1bitmc-rowcf']
param_grids['1bitMC-rowcf-PMF'] = {**param_grids['PMF'], **rowcf_grid, **one_bit_mc_grid}
param_grids['1bitMC-rowcf-PMF']['propensity_scores'] = ['1bitmc-rowcf']
param_grids['1bitMC-rowcf-SVDpp'] = {**param_grids['SVDpp'], **rowcf_grid, **one_bit_mc_grid}
param_grids['1bitMC-rowcf-SVDpp']['propensity_scores'] = ['1bitmc-rowcf']
param_grids['1bitMC-rowcf-SoftImpute'] = {**param_grids['SoftImpute'],
                                          **rowcf_grid, **one_bit_mc_grid}
param_grids['1bitMC-rowcf-SoftImpute']['propensity_scores'] = ['1bitmc-rowcf']
param_grids['1bitMC-rowcf-SoftImputeALS'] = {**param_grids['SoftImputeALS'],
                                       **rowcf_grid, **one_bit_mc_grid}
param_grids['1bitMC-rowcf-SoftImputeALS']['propensity_scores'] = ['1bitmc-rowcf']
param_grids['1bitMC-rowcf-MaxNorm'] = {**param_grids['MaxNorm'], **rowcf_grid,
                                      **one_bit_mc_grid}
param_grids['1bitMC-rowcf-MaxNorm']['propensity_scores'] = ['1bitmc-rowcf']
param_grids['1bitMC-rowcf-WTN'] = {**param_grids['WTN'], **rowcf_grid,
                                   **one_bit_mc_grid}
param_grids['1bitMC-rowcf-WTN']['propensity_scores'] = ['1bitmc-rowcf']

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
        'rowcf-PMF': WeightedSVD,
        'rowcf-SVD': WeightedSVD,
        'rowcf-SVDpp': WeightedSVDpp,
        'rowcf-SoftImpute': WeightedSoftImputeWrapper,
        'rowcf-SoftImputeALS': WeightedSoftImputeALSWrapper,
        'rowcf-MaxNorm': WeightedMaxNormWrapper,
        'rowcf-WTN': DoublyWeightedTraceNormWrapper,
        '1bitMC-rowcf-PMF': WeightedSVD,
        '1bitMC-rowcf-SVD': WeightedSVD,
        '1bitMC-rowcf-SVDpp': WeightedSVDpp,
        '1bitMC-rowcf-SoftImpute': WeightedSoftImputeWrapper,
        '1bitMC-rowcf-SoftImputeALS': WeightedSoftImputeALSWrapper,
        '1bitMC-rowcf-MaxNorm': WeightedMaxNormWrapper,
        '1bitMC-rowcf-WTN': DoublyWeightedTraceNormWrapper,
    }

# create a specific random seed per algorithm by hashing the algorithm name
algorithm_deterministic_seeds = \
    {algo_name: int(hashlib.sha256(algo_name.encode('utf-8')).hexdigest(), 16)
        % (2**32) for algo_name in algs_to_run}

# create dataset specific random seed for splitting cross-validation folds
dataset_deterministic_seed = \
    int(hashlib.sha256(dataset_name.encode('utf-8')).hexdigest(), 16) % (2**32)


# -----------------------------------------------------------------------------
# Construct training data
#

if dataset_name.startswith('ml-100k'):
    num = np.int(dataset_name.split('-')[-1])
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file(os.path.join('ml-100k',
                                               str(num), 'X_observed.txt'),
                                  reader=reader)

    # naive Bayes and logistic regression propensity score estimation are not
    # supported for this dataset
    for algo_name in algs_to_run:
        if algo_name.startswith('NB-') or algo_name.startswith('LR-'):
            algs_to_run.remove(algo_name)

elif dataset_name.startswith('binary-ml-100k'):
    num = np.int(dataset_name.split('-')[-1])
    reader = Reader(line_format='user item rating', sep='\t', rating_scale = (0,1))
    data = Dataset.load_from_file(os.path.join('binary-ml-100k',
                                               str(num), 'X_observed.txt'),
                                  reader=reader)

    # naive Bayes and logistic regression propensity score estimation are not
    # supported for this dataset
    for algo_name in algs_to_run:
        if algo_name.startswith('NB-') or algo_name.startswith('LR-'):
            algs_to_run.remove(algo_name)

elif dataset_name == 'coat':
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file('coat/train_surprise_format.csv',
                                  reader=reader)

    user_features = np.loadtxt('coat/user_item_features/user_features.ascii')
    item_features = np.loadtxt('coat/user_item_features/item_features.ascii')

    for algo_name in param_grids:
        if algo_name.startswith('LR-'):
            param_grids[algo_name]['propensity_scores'] = \
                ['logistic regression']
            param_grids[algo_name]['user_features'] = [user_features]
            param_grids[algo_name]['item_features'] = [item_features]

    # naive Bayes propensity score estimation is not supported for this dataset
    for algo_name in algs_to_run:
        if algo_name.startswith('NB-'):
            algs_to_run.remove(algo_name)

elif dataset_name.startswith('useritemfeature'):
    num = np.int(dataset_name.split('-')[1])
    reader = Reader(line_format='user item rating', sep='\t')
    data_dir = os.path.join('useritemfeature', str(num))
    data = Dataset.load_from_file(os.path.join(data_dir, 'X_observed.txt'),
                                  reader=reader)

    mcar_data = np.loadtxt(os.path.join(data_dir, 'MCAR_data.txt'))

    for algo_name in param_grids:
        if algo_name.startswith('NB-'):
            param_grids[algo_name]['propensity_scores'] = \
                ['naive bayes']
            param_grids[algo_name]['mcar_data'] = [mcar_data]

    user_features = np.loadtxt(os.path.join(data_dir, 'user_features.txt'))
    item_features = np.loadtxt(os.path.join(data_dir, 'item_features.txt'))

    for algo_name in param_grids:
        if algo_name.startswith('LR-'):
            param_grids[algo_name]['propensity_scores'] = \
                ['logistic regression']
            param_grids[algo_name]['user_features'] = [user_features]
            param_grids[algo_name]['item_features'] = [item_features]

elif dataset_name.startswith('steck'):
    num = np.int(dataset_name.split('-')[1])
    reader = Reader(line_format='user item rating', sep='\t')
    data_dir = os.path.join('steck', str(num))
    data = Dataset.load_from_file(os.path.join(data_dir, 'X_observed.txt'),
                                  reader=reader)

    mcar_data = np.loadtxt(os.path.join(data_dir, 'MCAR_data.txt'))

    for algo_name in param_grids:
        if algo_name.startswith('NB-'):
            param_grids[algo_name]['propensity_scores'] = \
                ['naive bayes']
            param_grids[algo_name]['mcar_data'] = [mcar_data]

    # logistic regression propensity score estimation is not supported for
    # this dataset
    for algo_name in algs_to_run:
        if algo_name.startswith('LR-'):
            algs_to_run.remove(algo_name)

elif dataset_name.startswith('rowcluster'):
    num = np.int(dataset_name.split('-')[1])
    reader = Reader(line_format='user item rating', sep='\t')
    data_dir = os.path.join('rowcluster', str(num))
    data = Dataset.load_from_file(os.path.join(data_dir, 'X_observed.txt'),
                                  reader=reader)

    mcar_data = np.loadtxt(os.path.join(data_dir, 'MCAR_data.txt'))

    for algo_name in param_grids:
        if algo_name.startswith('NB-'):
            param_grids[algo_name]['propensity_scores'] = \
                ['naive bayes']
            param_grids[algo_name]['mcar_data'] = [mcar_data]

    # logistic regression propensity score estimation is not supported for
    # this dataset
    for algo_name in algs_to_run:
        if algo_name.startswith('LR-'):
            algs_to_run.remove(algo_name)

elif dataset_name == 'yahoo':
    reader = Reader(line_format='user item rating', sep='\t')
    data = Dataset.load_from_file(
        'yahoo/ydata-ymusic-rating-study-v1_0-train.txt', reader=reader)
    train_set = data.build_full_trainset()

    reader = Reader(line_format='user item rating', sep='\t')
    test_data = Dataset.load_from_file(
        'yahoo/ydata-ymusic-rating-study-v1_0-test.txt', reader=reader)
    test_set = test_data.construct_testset(test_data.raw_ratings)

    yahoo_rng = np.random.RandomState(2244919967)
    mcar_data_sparse = \
        [test_set[idx]
         for idx in yahoo_rng.choice(len(test_set),
                                     int(len(test_set) * .05),
                                     replace=False)]

    mcar_data = np.zeros((train_set.n_users, train_set.n_items))
    for u, i, r in mcar_data_sparse:
        mcar_data[train_set.to_inner_uid(u),
                  train_set.to_inner_iid(i)] = r

    for algo_name in param_grids:
        if algo_name.startswith('NB-'):
            param_grids[algo_name]['propensity_scores'] = \
                ['naive bayes']
            param_grids[algo_name]['mcar_data'] = [mcar_data]

    # logistic regression propensity score estimation is not supported for
    # this dataset
    for algo_name in algs_to_run:
        if algo_name.startswith('LR-'):
            algs_to_run.remove(algo_name)
else:
    raise Exception('Dataset not supported: ' + dataset_name)

if dataset_name != 'yahoo':
    train_set = data.build_full_trainset()


# -----------------------------------------------------------------------------
# Hyperparameter search via cross-validation on training data
#

best_algs_rmse = {}
best_algs_mae = {}
grid_searches = {}

for algo_name in algs_to_run:
    print('[Dataset: %s - algorithm: %s - grid search CV]'
          % (dataset_name, algo_name), flush=True)
    kf = KFold(5, random_state=dataset_deterministic_seed)

    # for methods that estimate propensity scores and save them to a cache,
    # include the dataset name in the files generated for debug purposes
    if algo_name.startswith('NB-') or algo_name.startswith('LR-') \
            or algo_name.startswith('1bitMC-') \
            or algo_name.startswith('rowcf-'):
        param_grids[algo_name]['cache_prefix'] = ['%s_' % dataset_name]

    random.seed(algorithm_deterministic_seeds[algo_name])
    np.random.seed(algorithm_deterministic_seeds[algo_name])
    grid_search = GridSearchCV(algorithms[algo_name], param_grids[algo_name],
                               measures=['rmse', 'mae'], cv=kf,
                               n_jobs=num_jobs_for_cross_val)
    grid_search.fit(data)

    grid_searches[algo_name] = grid_search
    best_algs_rmse[algo_name] = grid_search.best_estimator['rmse']
    best_algs_mae[algo_name] = grid_search.best_estimator['mae']

    print('Best parameters found:', flush=True)
    best_params = copy.deepcopy(grid_searches[algo_name].best_params)
    if algo_name.startswith('NB-'):
        if 'mcar_data' in best_params['rmse']:
            del best_params['rmse']['mcar_data']
        if 'mcar_data' in best_params['mae']:
            del best_params['mae']['mcar_data']
    if algo_name.startswith('LR-'):
        if 'user_features' in best_params['rmse']:
            del best_params['rmse']['user_features']
        if 'item_features' in best_params['rmse']:
            del best_params['rmse']['item_features']
        if 'user_features' in best_params['mae']:
            del best_params['mae']['user_features']
        if 'item_features' in best_params['mae']:
            del best_params['mae']['item_features']
    print(json.dumps(best_params, indent=4, sort_keys=True), flush=True)
    print()


# -----------------------------------------------------------------------------
# Now train on full dataset per algorithm (using best hyperparameters found via
# cross-validation) and evaluate error on test data
#

if dataset_name.startswith('ml-100k'):
    num = np.int(dataset_name.split('-')[-1])
    reader = Reader(line_format='user item rating', sep='\t')
    test_data = Dataset.load_from_file(os.path.join('ml-100k',
                                                    str(num), 'X_test.txt'),
                                       reader=reader)
    test_set = test_data.construct_testset(test_data.raw_ratings)
elif dataset_name.startswith('binary-ml-100k'):
    num = np.int(dataset_name.split('-')[-1])
    reader = Reader(line_format='user item rating', sep='\t', rating_scale = (0,1))
    test_data = Dataset.load_from_file(os.path.join('binary-ml-100k',
                                                    str(num), 'X_test.txt'),
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
                                                    str(num), 'S.txt'),
                                       reader=reader)
    test_set = test_data.construct_testset(test_data.raw_ratings)
elif dataset_name.startswith('steck'):
    num = np.int(dataset_name.split('-')[1])
    reader = Reader(line_format='user item rating', sep='\t')
    test_data = Dataset.load_from_file(os.path.join('steck', str(num),
                                                    'S.txt'),
                                       reader=reader)
    test_set = test_data.construct_testset(test_data.raw_ratings)
elif dataset_name.startswith('rowcluster'):
    num = np.int(dataset_name.split('-')[1])
    reader = Reader(line_format='user item rating', sep='\t')
    test_data = Dataset.load_from_file(os.path.join('rowcluster', str(num),
                                                    'S.txt'),
                                       reader=reader)
    test_set = test_data.construct_testset(test_data.raw_ratings)
elif dataset_name == 'yahoo':
    pass


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

    if algo_name.startswith('NB') or algo_name.startswith('LR') \
            or algo_name.startswith('rowcf') or algo_name.startswith('1bitMC'):
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

    if algo_name.startswith('NB') or algo_name.startswith('LR') \
            or algo_name.startswith('rowcf') or algo_name.startswith('1bitMC'):
        propensity_estimates = algo.propensity_estimates
        np.savetxt(os.path.join(output_dir,
                                '%s_%s_mae_propensity_estimates.txt'
                                % (dataset_name, algo_name)),
                   propensity_estimates)

    print(algo_name,
          'MAE:', accuracy.mae(predictions, verbose=False),
          flush=True)

print()
