import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from scipy.special import expit
import pickle

PICKLE_FILE = 'Model/picket_data.pickle'
with open(PICKLE_FILE, 'rb') as f:
    PICKET_DATA = pickle.load(f)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, sequence_len, feature_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feature_n).repeat(1, sequence_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2 * self.hidden_size)
        
        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        attention_weights = self.to_weight(x).view(batch_size, sequence_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

class EncoderNet(nn.Module):
    def __init__(self, input_size=4096, hidden_size=512, dropout_rate=0.2):
        super().__init__()
        self.compress = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_data):
        batch_size, seq_len, feat_n = input_data.size()
        data_in = input_data.view(-1, feat_n)
        data_in = self.compress(data_in)
        data_in = self.dropout(data_in)
        data_in = data_in.view(batch_size, seq_len, -1)
        output, (hidden_state, _) = self.lstm(data_in)
        return output, hidden_state

class DecoderNet(nn.Module):
    def __init__(self, hidden_size=512, word_dim=1024, dropout_rate=0.3):
        super().__init__()
        self.vocab_size = len(PICKET_DATA) + 4
        self.output_size = self.vocab_size
        self.hidden_size = hidden_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(self.vocab_size, word_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, self.output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        batch_size = encoder_last_hidden_state.size(1)
        decoder_hidden = encoder_last_hidden_state
        decoder_context = torch.zeros_like(decoder_hidden)
        decoder_input = Variable(torch.ones(batch_size, 1)).long().to(encoder_last_hidden_state.device)

        seq = []
        seq_predictions = []

        targets = self.embedding(targets) if targets is not None else None
        seq_len = targets.size(1) if targets is not None else 28

        for i in range(seq_len - 1):
            if mode == 'train':
                threshold = self._teacher_forcing_ratio(tr_steps)
                use_teacher_forcing = random.random() < threshold
                current_input_word = targets[:, i] if use_teacher_forcing else self.embedding(decoder_input).squeeze(1)
            else:
                current_input_word = self.embedding(decoder_input).squeeze(1)

            context = self.attention(decoder_hidden, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, (decoder_hidden, decoder_context) = self.lstm(lstm_input, (decoder_hidden, decoder_context))
            
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq.append(logprob.unsqueeze(1))
            decoder_input = logprob.max(1)[1].unsqueeze(1)

        seq = torch.cat(seq, dim=1)
        seq_predictions = seq.max(2)[1]
        return seq, seq_predictions

    def _teacher_forcing_ratio(self, training_steps):
        if training_steps is None:
            return 0
        return expit(training_steps / 20 + 0.85)

class ModelMain(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        encoder_output, encoder_last_hidden_state = self.encoder(avi_feat)
        
        if mode == 'train':
            return self.decoder(encoder_last_hidden_state, encoder_output, target_sentences, mode, tr_steps)
        elif mode == 'inference':
            return self.decoder(encoder_last_hidden_state, encoder_output)
        else:
            raise ValueError(f"Invalid mode: {mode}. Expected 'train' or 'inference'.")