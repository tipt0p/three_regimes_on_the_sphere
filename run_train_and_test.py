#! /usr/bin/env python
import os
import subprocess
import argparse
import datetime
import time

parser = argparse.ArgumentParser(description='Managing experiments')
parser.add_argument('--test', action='store_true',
                    help='print (test) or os.system (run)')

args = parser.parse_args()

if args.test:
    action = print
else:
    action = os.system

ENVIRONMENT = 'SI_regimes_env'

data = 'CIFAR10' # 'CIFAR10' 'CIFAR100' 
network = 'ConvNetSI'# 'ConvNetSI' 'ConvNetSIAf' 'ResNet18SI' 'ResNet18SIAf'
fix_elr = 'fix_elr' # 'fix_elr' 'fix_lr'
fix_noninvlr = 0.0 # 0.0 to fix non-scale-invariant parameters, -1 to train them
same_last_layer = True # init the last layer with the same seed for all exps
add_params = ''

save_path = './Experiments'
if not os.path.exists(save_path):
    os.mkdir(save_path)

save_path = save_path + '/' + fix_elr
if fix_elr == 'fix_elr':
    add_params += '--fix_elr '
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
if same_last_layer:
    save_path = save_path + '/same_last_layer'
    add_params += '--same_last_layer '
if not os.path.exists(save_path):
    os.mkdir(save_path)

if fix_noninvlr >= 0:
    save_path = save_path + '/{}_{}_noninvlr_{}/'.format(network, data, fix_noninvlr)
else:
    save_path = save_path + '/{}_{}/'.format(network, data)
if not os.path.exists(save_path):
    os.mkdir(save_path)

params = {'dataset': data, # dataset
          'model': network, # network architecture
          'lr_init': 0.0, # learning rate, use it to train networks in the whole parameter space
          'elr': 1e-3, # elr, use it to train networks on the sphere (with fix_elr flag!!!)
          'noninvlr': fix_noninvlr, # lr for non-scale-invariant parameters
          'wd': 0.0, # weight decay
          'momentum': 0.0, # momentum
          'num_channels': 32, # network width
          'depth': 3, # network depth
          'epochs': 3, # number of epochs
          'corrupt_train': 0.0, # label noise
          'save_freq': 1, # save checkpoint each x epochs
          'eval_freq': 1000, # write test metrics into the log each x epochs (we do not use them)
          'use_data_size': 50000, # use all training data
          'dir': save_path + 'checkpoints', # save checkpoints here
          'init_scale': 10, # scale of the last layer init
          'gpu': 0, # GPU
          'seed': 1 # seed
          }

add_params += '--use_test --no_schedule --no_aug'
# --no_schedule      - train without learning rate schedule
# --no_aug           - train without data augmentation
# --cosan_schedule   - train with cosine learning rate schedule 

params_test = {'dataset': data,
               'model': network,
               'num_channels': params['num_channels'],
               'depth': params['depth'],
               'init_scale': params['init_scale'],
               'save_path': save_path + 'info',
               'models_dir': save_path + 'checkpoints',
               'use_data_size': params['use_data_size'],
               'gpu': params['gpu']
               }

log_path = save_path + 'logs/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(log_path):
    os.mkdir(log_path)

info_path = save_path + 'info/'
if not os.path.exists(info_path):
    os.mkdir(info_path)

commands = []

if fix_elr == 'fix_elr':
    exp_name = 'c{}_d{}_elr{}_epoch{}'.format(params['num_channels'], params['depth'], 
                                              params['elr'], params['epochs'])
else:
    exp_name = 'c{}_d{}_lr{}_wd{}_epoch{}'.format(params['num_channels'], params['depth'], 
                                              params['lr_init'],params['wd'], params['epochs'])

if 'no_schedule' in add_params:
    exp_name = exp_name + '_nosch'
if params['init_scale'] > 0:
    exp_name = exp_name + 'initscale{}'.format(params['init_scale'])
if 'no_aug' in add_params:
    exp_name = exp_name + '_noaug'
if 'cosan_schedule' in add_params:
    exp_name = exp_name + '_cosan'
if params['seed'] > 1:
    exp_name = exp_name + '_seed{}'.format(params['seed'])
if params['corrupt_train'] > 0:
    exp_name = exp_name + '_corr{}'.format(params['corrupt_train'])
if params['momentum'] > 0:
    exp_name = exp_name + '_mom{}'.format(params['momentum'])

params['dir'] = params['dir'] + '/' + exp_name
exp_log_path = log_path + exp_name

params_test['models_dir'] = params_test['models_dir'] + '/' + exp_name + '/trial_0'

if params['corrupt_train'] > 0:
    subsets = ' --eval_on_train_subsets '
else:
    subsets = ''


# train
commands.append(
    'train.py {} >> {}'.format(' '.join(["--{} {}".format(k, v) for (k, v) in params.items()]) + ' ' + add_params,
                               exp_log_path + '.out'))

p_test = params_test.copy()
# train info
p_test['save_path'] = params_test['save_path'] + '/' + exp_name + '/train-tm.npz'
commands.append('get_info.py {} --corrupt_train {} --train_mode {} --eval_model --all_pnorm'.format(
    ' '.join(["--{} {}".format(k, v) for (k, v) in p_test.items()]), params['corrupt_train'], subsets))
#add --calc_grad_norms to compute gradient norms in case of non SI nets

# test info
p_test['save_path'] = params_test['save_path'] + '/' + exp_name + '/test-em.npz'
commands.append('get_info.py {} --use_test --eval_model'.format(
    ' '.join(["--{} {}".format(k, v) for (k, v) in p_test.items()])))

# prebn info
p_test['save_path'] = params_test['save_path'] + '/' + exp_name + '/train-tm-prebn.npz'
commands.append('get_info.py {} --corrupt_train {} --train_mode {} --all_pnorm --calc_grad_norms --prebn_only'.format(' '.join(["--{} {}".format(k,v) for (k, v) in p_test.items()]), params['corrupt_train'], subsets))
#add --calc_grad_norms to compute gradient norms in case of SI nets

if ENVIRONMENT:
    tmp_str = ' && ~/anaconda3/envs/{}/bin/python '.format(ENVIRONMENT)
    final_command = "bash -c '. activate {} {} {}'".format(ENVIRONMENT, tmp_str, tmp_str.join(commands))
else:
    final_command = 'python '.join(commands)

action(final_command)
