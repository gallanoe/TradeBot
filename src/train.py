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
from model import *
from data import *

def train(model, loader, optimizer, criterion, device):
	model.to(device)
	model.train()
	epoch_loss = 0 
	for i, (input, target) in enumerate(loader):
		input, target = input.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(input, target)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
	return epoch_loss / len(loader)

def eval(model, loader, criterion, device):
	model.to(device)
	model.eval()
	epoch_loss = 0
	for i, (input, target) in enumerate(loader):
		input, target = input.to(device), target.to(device)
		output = model(input, target)
		loss = criterion(output, target)
		epoch_loss += loss.item()
	return epoch_loss / len(loader)

def time_elapsed(start_time, end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time) / 60
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60)) 
	return elapsed_mins, elapsed_secs2
