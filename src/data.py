import yfinance as yf
import numpy as np 
import pandas as pd 
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
import os 
import pathlib
import csv
import matplotlib.pyplot as plt 
import torch
from torch import nn 
from torch import optim
from torch.nn import functional as F
from features import *

## Design philosophy
# Store raw data into dct
# Create windows and create normalized windows, store into dct as well
# Allow deletion of keys in dct (input, target pairs)
# Turn dct into datasets

def download_data(symbols, funcs, period='max', interval='1d'): 
	"""
		Downloads ticker history over period of 'period' and interval of 'interval' 
		from yahoo finance and applies the functions listed in 'funcs' to
		the downloaded DataFrames.
	"""
	total_entries = 0 
	dct = {}
	for symbol in symbols:
		df = yf.Ticker(symbol).history(period=period, interval=interval)
		df = df.drop(columns=['Dividends', 'Stock Splits'])
		for func in funcs:
			func(df)
		df = df.replace(np.inf, np.nan)
		df = df.dropna() 
		total_entries += df.shape[0]
		print(f'{df.shape[0]} data points for {symbol}')
		dct[symbol]['raw_data'] = df
	print(f'{total_entries} downloaded')
	return dct

def create_windows(dct, time_steps_before, time_steps_after, step_size,
											 input_cols=None, target_cols=['Close'], transform=False,
											 groups=None, print_every=500):
	count = 0
	for symbol in dct:
		df = dct[symbol]['raw_data']
		inputs, targets, count = _create_windows(df, time_steps_before, time_steps_after, step_size, 
																		 input_cols=input_cols, target_cols=target_cols, transform=transform,
																		 groups=groups, print_every=print_every, count=count)
		if transform: 
			dct[symbol]['norm_inputs'] = inputs
			dct[symbol]['norm_targets'] = targets
		else:
			dct[symbol]['inputs'] = inputs
			dct[symbol]['targets'] = targets

	return dct


def _create_windows(df, time_steps_before, time_steps_after, step_size, 
	                    input_cols=None, target_cols=['Close'], transform=False,
	                    groups=None, print_every=500, count=0):

	inputs, targets = [], []
	for i in range(0, len(df) - time_steps_before - time_steps_after + 1, step_size):
		combined = df.iloc[i:i + time_steps_before + time_steps_after].copy()
		input = combined.iloc[:time_steps_before]
		target = combined.iloc[time_steps_before:]
		count += 1

		if i % print_every == print_every - 1:
			print(f'Processed {i+1} items...')
	return input, target, count


# def create_windows(dct, time_steps_before, time_steps_after, step_size,
# 									 input_cols=None, target_cols=['Close'], transform=False, groups=None, print_every=500):
# 	# For list of dfs case
# 	if type(dct) is dict:
# 		num_windows = 0
# 		dct = {} 
# 		for sym in dfs:
# 			df = dfs[sym] 
# 			dct[sym] = {}
# 			tup = create_windows(df, time_steps_before, time_steps_after, step_size,
# 																input_cols, target_cols, transform, groups, print_every)
# 			dct[sym]['inputs'] = tup[0]
# 			dct[sym]['targets'] = tup[1]
# 			num_windows += len(tup[0])
# 		print(f'{num_windows} windows created')
# 		return dct

# 	# For single df case
# 	df = dfs

# 	inputs, targets = [], []
# 	for i in range(0, len(df) - time_steps_before - time_steps_after + 1, step_size):
# 		combined = df.iloc[i:i + time_steps_before + time_steps_after].copy()
# 		input = combined.iloc[:time_steps_before]
# 		target = combined.iloc[time_steps_before:]

# 		if transform:
# 			assert groups is not None, "'groups' argument cannot be none when 'transform' argument is True, must be list of list of strings"
# 			input, target = normalize_pair(input, target, groups)

# 		if input_cols is not None:
# 			inputs.append(input[input_cols])
# 		else:
# 			inputs.append(input)

# 		if target_cols is not None:
# 			targets.append(target[target_cols])
# 		else:
# 			targets.append(target)

# 		if i % print_every == print_every - 1:
# 			print(f'Processed {i+1} items...')
# 	return inputs, targets


def filter_columns(dct, input_columns, target_columns):
	for symbol in dct:
		for key in dct[symbol]:
			if 'input' in key:
				dct[symbol][key] = dct[symbol][key][input_columns]
			if 'target' in key:
				dct[symbol][key] = dct[symbol][key][target_columns]

	return dct

def


def normalize_data(dct, groups=None, combine=False, inplace=False):
	"""
		Applies the normalization technique specified in `normalize_pair` to all the windows
		in the dictionary of data, passing the `groups` and `combine` parameters to calls of 
		`normalize_pair`. This function modified the passed dictionary by adding the `norm_inputs`
		and `norm_targets` key for each symbol dictionary
	"""

	for symbol in dct:

		dct[symbol]['norm_inputs'], dct_symbol['norm_targets'] = [], []

		input_dfs, target_dfs = dct[symbol]['inputs'], dct[symbol]['targets']
		for input, target in zip(input_dfs, target_dfs):
			norm_input, norm_target = normalize_pair(input, target, groups=groups, combine=combine)
			dct[symbol]['norm_inputs'].append(norm_input)
			dct[symbol]['norm_targets'].append(norm_target)

	return dct


def normalize_pair(input, target, groups=None, combine=False):
	"""
		Given a pair of DataFrame objects serving as corresponding input/target pairs and
		a list of groups, normalizes the data group-wise; that is, the scaler will measure empirical
		means and variances over multiple feature columns as it would a single feature column. 
		If groups is None, the scaler will normalize each feature independently, as it normally would. 
		If combine is False, the scaler will fit only the input data. If combine is True, the scaler will fit both
		the input and target data.
	"""
	if groups is None:

		scaler = sklearn.preprocessing.StandardScaler()

		if combine:
			combined = np.concatenate((input.vaues, target.values), axis=0)
			scaler.fit(combined)
		else:
			scaler.fit(input)

		input[input.columns] = scaler.transform(input[input.columns])
		target[target.columns] = scaler.transform(target[target.columns])

		return input, target

	for group in groups:

		input_group, target_group = input[group].copy(), target[group].copy()
		input_shape, target_shape = input_group.shape, target_group.shape
		input_group, target_group = input_group.values.reshape(-1, 1), target_group.values.reshape(-1, 1)

		scaler = sklearn.preprocessing.StandardScaler()

		if combine:
			combined = np.concatenate((input_group, target_group), axis=0)
			scaler.fit(combined)
		else:
			scaler.fit(input_group)

		input.loc[:,group] = scaler.transform(input_group).reshape(*input_shape)
		target.loc[:,group] = scaler.transform(target_group).reshape(*target_shape)

	return input, target


def save_data(dct):
	"""
		Given a dictionary of the following format (usually returned by the create_windows function):
			dct[symbol] is dictionary
			dct[symbol]['inputs'] is list of DataFrames
			dct[symbol]['targets'] is list of DataFrames
		the function will save the data organized in this manner:
			-> data
				-> symbol
					-> input 
						-> symbol_input_01.csv
					-> target 
						-> symbol_target_01.csv
		where the .csv files correspond to a DataFrame in the list of DataFrames specified in the
		arrangement of the dictionary above.
	"""
	# Make data directory
	if not os.path.exists('./data'):
		os.mkdir('./data')

	for symbol in dct:
		# Make symbol directory 
		path = f'./data/{symbol}'
		if not os.path.exists(path):
			os.mkdir(path)
		for key in dct[symbol]:

		if not os.path.exists(path+'/input'):
			os.mkdir(path+'/input')
		if not os.path.exists(path+'/target'):
			os.mkdir(path+'/target')

		inputs, targets = dct[symbol]['inputs'], dct[symbol]['targets']
		assert len(inputs) == len(targets), "# inputs should equal # targets"

		for i, (input, target) in enumerate(zip(inputs, targets)):
			input_fname = f'{symbol}_input_' + str(i).zfill(4)
			target_fname = f'{symbol}_target_' + str(i).zfill(4)

			input.to_csv(rf'./data/{symbol}/input/{input_fname}.csv')
			target.to_csv(rf'./data/{symbol}/target/{input_fname}.csv')


def load_data(symbols):

	if not os.path.exists('./data'):
		print("Data directory does not exist. Call 'download_data' followed by 'save_data' to cache processed data onto hard drive")
		return 
	dct = {}
	num_windows = 0

	for symbol in symbols:

		dct_entry = {}

		input_path = f'./data/{symbol}/input'
		target_path = f'./data/{symbol}/target'

		if not os.path.exists(f'./data/{symbol}'):
			print(f"Data directories for {symbol} does not exist.")
			continue 
		if not os.path.exists(input_path) or not os.path.exists(target_path):
			print(f"Data directories for {symbol} does not exist.")
			continue 

		input_files = os.listdir(input_path)
		target_files = os.listdir(target_path)

		input_files.sort()
		target_files.sort()

		dct_entry['inputs'], dct_entry['targets'] = [], []
		for input_file, target_file in zip(input_files, target_files):
			dct_entry['inputs'].append(pd.read_csv(input_path + '/' + input_file, index_col='Date'))
			dct_entry['targets'].append(pd.read_csv(target_path + '/' + target_file, index_col='Date'))

		num_windows += len(dct_entry['inputs'])
		dct[symbol] = dct_entry

	print(f'Loaded {num_windows} windows')
	return dct


def create_datasets(dct, valid_pct=0.2):

	train_inputs, train_targets = None, None
	valid_inputs, valid_targets = None, None
	for symbol in dct:
		assert len(dct[symbol]['inputs']) == len(dct[symbol]['targets'])
		num_points = len(dct[symbol]['inputs'])
		print(f'{symbol} has {num_points} input-target pairs')
		idxs = np.arange(len(dct[symbol]['inputs']), dtype=np.int32)
		train_idxs, valid_idxs = sklearn.model_selection.train_test_split(idxs, test_size=valid_pct)

		inputs, targets = dct[symbol]['inputs'], dct[symbol]['targets']
		inputs = np.array([input.values for input in inputs])
		targets = np.array([target.values for target in targets])

		train_inputs_to_add, train_targets_to_add = inputs[train_idxs], targets[train_idxs]
		valid_inputs_to_add, valid_targets_to_add = inputs[valid_idxs], targets[valid_idxs]

		if train_inputs is None:
			train_inputs = train_inputs_to_add
		else:
			train_inputs = np.concatenate((train_inputs, train_inputs_to_add), axis=0)

		if train_targets is None:
			train_targets = train_targets_to_add
		else:
			train_targets = np.concatenate((train_targets, train_targets_to_add), axis=0)

		if valid_inputs is None:
			valid_inputs = valid_inputs_to_add
		else:
			valid_inputs = np.concatenate((valid_inputs, valid_inputs_to_add), axis=0)

		if valid_targets is None:
			valid_targets = valid_targets_to_add
		else:
			valid_targets = np.concatenate((valid_targets, valid_targets_to_add), axis=0)

	train_dataset = torch.utils.data.TensorDataset(
		torch.tensor(train_inputs).float(),
		torch.tensor(train_targets).float()
	)
	valid_dataset = torch.utils.data.TensorDataset(
		torch.tensor(valid_inputs).float(),
		torch.tensor(valid_targets).float()
	)
	return train_dataset, valid_dataset

def create_loaders(train_dataset, valid_dataset, batch_size=64, jobs=0):
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=jobs)
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=jobs)
	return train_dataloader, valid_dataloader
