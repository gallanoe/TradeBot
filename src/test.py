from data import *
from features import *
from model import *
from train import *
import time

FUNCS = [MACD, CCI, MA20TR, MA, BOLL, MTM, ROC, SMI, WVAD, RSI, RET, VOL]
SYMBOLS= ['AAPL', 'AMZN', 'GOOGL', 'MSFT', 'FB', 'INTC', 'AMD', 'NVDA', 'ADBE', 'T', 'NFLX', 'SAP', 'ORCL', 'IBM',
           'CRM', 'VMW', 'INTU', 'ADSK', 'PYPL', 'SQ', 'SNAP', 'TWTR', 'DELL', 'HPQ']
GROUPS = [
  ['Open', 'High', 'Low', 'Close',
   'MA5', 'MA10', 'EMA20',
   'BOLU', 'BOLD',
   'VOL1', 'VOL5', 'VOL10', 'VOL20'],
  ['Volume'],
  ['MACD'],
  ['CCI'],
  ['MA20TR'],
  ['MTM6', 'MTM12', 'ROC'],
  ['SMI'],
  ['WVAD'],
  ['RSI'],
  ['RET1', 'RET5', 'RET10', 'RET20']
]
N_EPOCHS = 100
MODEL_NAME = 'test'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


## LOAD 
# dfs = download_data(SYMBOLS, FUNCS, period='5y', interval='1d')
# dct = create_windows(dfs, time_steps_before=40, time_steps_after=10, step_size=5)
# save_data(dct)


## TRAIN
start_time = time.time()

dct = load_data(SYMBOLS)
dct = normalize_data(dct, groups=GROUPS)
train_dataset, valid_dataset = create_datasets(dct)
train_dataloader, valid_dataloader = create_loaders(train_dataset, valid_dataset)

# encoder = EncoderRNN()
# decoder = DecoderRNN()
# model = Seq2Seq(encoder, decoder).to(device)
# 
# best_valid_loss = float('inf')
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.MSELoss()
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
# 
# train_losses = []
# valid_losses = []
# 
# for epoch in range(N_EPOCHS):
	# epoch_start_time = time.time()
# 
	# train_loss = train(model, train_dataloader, optimizer, criterion, device)
	# valid_loss = train(model, valid_dataloader, criterion, device)	
# 
	# train_losses.append(train_loss)
	# valid_losses.append(valid_loss)
	# scheduler.step()
# 
	# epoch_end_time = time.time()
# 
	# if valid_loss < best_valid_loss:
		# best_valid_loss = valid_loss
		# torch.save(model.state_didct(), MODEL_NAME+'.pt')
# 
	# epoch_mins, epoch_secs = time_elapsed(epoch_start_time, epoch_end_time)
# 
	# print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train loss: {train_loader:.3f} | Valid loss: {valid_loss:.3f}')
# 
print(train_loader[0])
# 
# end_time = time.time()
# mins, secs == time_elapsed(start_time, end_time)
# print(f'Total time: {mins}m {secs}s')
