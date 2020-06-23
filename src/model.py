import torch
from torch import nn 
from torch import optim
from torch.nn import functional as F

class EncoderRNN(nn.Module):
  def __init__(self, input_size=27, hidden_size=512, bidirectional=False, dropout=0.2, num_layers=6):
    super(EncoderRNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    self.num_directions = 2 if self.bidirectional else 1
    self.dropout = dropout
    self.lstm = nn.LSTM(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        bidirectional=self.bidirectional,
        batch_first=True,
        dropout=self.dropout
    )

  def forward(self, input, hidden):
    output, hidden = self.lstm(input, hidden)
    return output, hidden

  def init_hidden(self, batch_size):
    return (torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device))


class DecoderRNN(nn.Module):
  def __init__(self, input_size=1, hidden_size=512, output_size=1, bidirectional=False, dropout=0.2, num_layers=6):
    super(DecoderRNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    self.num_directions = 2 if self.bidirectional else 1
    self.dropout = dropout
    self.lstm = nn.LSTM(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        bidirectional=self.bidirectional,
        batch_first=True,
        dropout=self.dropout
    )
    self.out = nn.Linear(self.num_directions*self.hidden_size, self.output_size)

  def forward(self, input, hidden):
    output, hidden = self.lstm(input, hidden)
    output = self.out(output)
    return output, hidden

  def init_hidden(self, batch_size):
    return (torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device),
            torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size, device=device))


class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2Seq, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    assert self.encoder.hidden_size == self.decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"

  def forward(self, input, target, teacher_forcing_ratio=0.5):
    input_batch, input_len, input_features = input.shape
    target_batch, target_len, target_features = target.shape

    outputs = torch.zeros(target_batch, target_len, target_features).to(device)

    encoder_hidden = self.encoder.init_hidden(input_batch)
    _, encoder_hidden = self.encoder(input, encoder_hidden)

    input = input[:,-1,3].unsqueeze(1).unsqueeze(1)
    decoder_hidden = encoder_hidden

    for t in range(target_len):
      output, decoder_hidden = self.decoder(input, decoder_hidden)
      outputs[:,t:t+1,:] = output

      teacher_forcing = random.random() < teacher_forcing_ratio
      input = target[:,t:t+1,:] if teacher_forcing else output

    return outputs