import yfinance
import numpy as np 
import pandas as pd 
import sklearn
from sklearn import preprocessing
from sklearn import model_selection
import os 
import pathlib
import csv
import matplotlib.pyplot as plt 
from features import *

def download_data(symbols, funcs, period='max', interval='1d'): 
	dfs = {}
	for symbol in symbols:
		df = yf.Ticker(symbol).history(period=period, interval=interval)
		for func in funcs:
			func(df)
		df = df.replace(np.inf, np.nan)
		df = df.dropna() 
		dfs[symbol] = df
	return dfs

def create_windows(dfs, time_steps_before, time_steps_after, step_size,
									 Xcols=None, ycols=['Close'], transform=False, groups=None, print_every=500):
	# For list of dfs case
	if type(dfs) is dict:
		dct = {} 
		for sym in dfs:
			df = dfs[sym] 
			dct[sym] = {}
			tup = create_windows(df, time_steps_before, time_steps_after, step_size,
																Xcols, ycols, transform, groups, print_every)
			dct[sym]["Xdfs"] = tup[0]
			dct[sym]["ydfs"] = tup[1]
		return dct

	# For single df case
	df = dfs

	Xs, ys = []
	Xdfs, ydfs = [], []
	for i in range(0, len(df) - time_steps_before - time_steps_after + 1, step_size):
		Xyi = df.iloc[i:i + time_steps_before + time_steps_after].copy()

		if transform:
			assert groups is not None, "'groups' argument cannot be none when 'transform' argument is True, must be list of list of strings"
			Xyi = normalize_window(Xyi, groups)

		Xi = Xyi.iloc[:time_steps_before]
		yi = Xyi.iloc[time_steps_before:]

		if Xcols is not None:
			Xdfs.append(Xi[Xcols])
		else:
			Xdfs.append(Xi)

		if ycols is not None:
			ydfs.append(yi[ycols])
		else:
			ydfs.append(yi)

		if i % print_every == print_every - 1:
			print(f'Processed {i+1} items...')
	return Xdfs, ydfs

def normalize_window(Xydf, groups):
	for group in groups:
		Xydfg = Xydf[group].copy()
		Xydfg_shp = Xydfg.shape
		Xydfg = Xydfg.values.reshape(-1,1)
		scaler = sklearn.preprocessing.StandardScaler()
		scaler.fit(Xydf.iloc[:time_steps_before][group].values.reshape(-1,1))
		Xydf.loc[:,group] = scaler.transform(Xydfg).reshape(*Xydfg_shp)
	return Xydf

def save_data(dct):
	# Make data directory
	if not os.path.exists('./data'):
		os.mkdir('./data')
	for sym in dct:
		# Make symbol directory 
		if not os.path.exists('./data/{}'.format(sym)):
			os.mkdir('./data/{}'.format(sym))
		data = dct[sym]
		Xdfs, ydfs = data["Xdfs"], data["ydfs"]
		assert len(Xdfs) == len(ydfs), "# inputs should equal # targets"
		for i, (Xdf, ydf) in enumerate(zip(Xdfs, ydfs)):
			Xdf.to_csv(r'./data/{}/input/{}.csv'.format(sym, i))
			ydf.to_csv(r'./data/{}/target/{}.csv'.format(sym, i))

def load_data():
	dct = {}
	
	pass

def windows_to_array(windows):
	return np.array([windows.values for window in windows])

def create_datasets(Xs, ys, valid_pct=0.2):
	assert len(Xs) == len(ys)
	idxs = np.arange(len(Xs), dtype=np.int32)
	train_idxs, valid_idxs = sklearn.model_selection.train_test_split(idxs, test_size=valid_pct)
	train_Xs, train_ys = Xs[train_idxs], ys[train_idxs]
	valid_Xs, valid_ys = Xs[valid_idxs], ys[valid_idxs]
	print(f'Train size: {len(train_idxs)}, Valid size: {len(valid_idxs)}')
  trn_ds = torch.utils.data.TensorDataset(
    torch.tensor(Xs[trn_idxs]).float(),
    torch.tensor(ys[trn_idxs]).float()
  )
  val_ds = torch.utils.data.TensorDataset(
    torch.tensor(Xs[val_idxs]).float(),
    torch.tensor(ys[val_idxs]).float()
  )

def create_loaders(train_dataset, valid_dataset, batch_size=64, jobs=0):
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=jobs)
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=jobs)
	return train_dataloader, valid_dataloader
