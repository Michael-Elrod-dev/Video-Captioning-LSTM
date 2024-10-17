import os
import re
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import model

DATA_PATH = 'MLDS_hw2_1_data/'
MODEL_PATH = 'Model'
PICKLE_FILE = 'Model/picket_data.pickle'

class DataProcessor(Dataset):
    def __init__(self, label_file, files_dir, dictionary, words_to_index):
        self.label_file = label_file
        self.files_dir = files_dir
        self.avi = read_files(label_file)
        self.words_to_index = words_to_index
        self.dictionary = dictionary
        self.data_pair = process_captions(files_dir, words_to_index)

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        assert idx < self.__len__()
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000) / 10000
        return torch.Tensor(data), torch.Tensor(sentence)

class TestDataLoader(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])

    def __len__(self):
        return len(self.avi)

    def __getitem__(self, idx):
        return self.avi[idx]

def create_dictionary(word_min):
    with open(os.path.join(DATA_PATH, 'training_label.json'), 'r') as f:
        file = json.load(f)
    
    word_count = {}
    for d in file:
        for s in d['caption']:
            words = re.sub(r'[.!,;?\]]', ' ', s).split()
            for word in words:
                word = word.replace('.', '') if '.' in word else word
                word_count[word] = word_count.get(word, 0) + 1

    dictionary = {word: count for word, count in word_count.items() if count > word_min}
    
    special_tokens = [('<PAD>', 0), ('< SOS >', 1), ('<EOS>', 2), ('<UNK>', 3)]
    index_to_word = {i + len(special_tokens): w for i, w in enumerate(dictionary)}
    words_to_index = {w: i + len(special_tokens) for i, w in enumerate(dictionary)}
    
    for token, index in special_tokens:
        index_to_word[index] = token
        words_to_index[token] = index
    
    return index_to_word

def process_sentence(sentence, words_to_index):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    processed = [words_to_index.get(word, words_to_index['<UNK>']) for word in sentence]
    return [words_to_index['< SOS >']] + processed + [words_to_index['<EOS>']]

def process_captions(label_file, words_to_index):
    with open(label_file, 'r') as f:
        label = json.load(f)
    
    annotated_caption = []
    for d in label:
        for s in d['caption']:
            processed = process_sentence(s, words_to_index)
            annotated_caption.append((d['id'], processed))
    return annotated_caption

def read_files(files_dir):
    avi_data = {}
    files = os.listdir(files_dir)
    for file in files:
        value = np.load(os.path.join(files_dir, file))
        avi_data[file.split('.npy')[0]] = value
    return avi_data

def train(model, epoch, train_loader, loss_func):
    model.train()
    print(f"Epoch: {epoch}")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    running_loss = 0.0

    for batch_idx, (avi_feats, ground_truths, lengths) in enumerate(train_loader):
        avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

        optimizer.zero_grad()
        seq_logProb, _ = model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)

        ground_truths = ground_truths[:, 1:]
        loss = calculate_loss(seq_logProb, ground_truths, lengths, loss_func)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch: {batch_idx+1}/{len(train_loader)}, Loss: {running_loss/10:.3f}")
            running_loss = 0.0

    print(f'Epoch: {epoch} & loss: {loss.item():.3f}')

def test(test_loader, model, index_to_word):
    model.eval()
    results = []
    for id, features in test_loader:
        features = Variable(features).float()
        _, seq_predictions = model(features, mode='inference')
        predictions = [[index_to_word.get(x.item(), 'something') for x in s] for s in seq_predictions]
        captions = [' '.join(s).split('<EOS>')[0] for s in predictions]
        results.extend(zip(id, captions))
    return results

def calculate_loss(predictions, targets, lengths, loss):
    p_cat = torch.cat([pred[:length-1] for pred, length in zip(predictions, lengths)])
    t_cat = torch.cat([target[:length-1] for target, length in zip(targets, lengths)])
    loss = loss(p_cat, t_cat)
    return loss / len(predictions)

def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data)
    avi_data = torch.stack(avi_data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths

def main():
    index_to_word, words_to_index, dictionary = create_dictionary(4)
    
    train_dataset = DataProcessor(
        os.path.join(DATA_PATH, 'training_data/feat'),
        os.path.join(DATA_PATH, 'training_label.json'),
        dictionary, words_to_index
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=32, shuffle=True,
        num_workers=4, collate_fn=minibatch
    )
    
    epochs_n = 20
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(index_to_word, f)

    loss = nn.CrossEntropyLoss()
    encoder = model.EncoderNet()
    decoder = model.DecoderNet(hidden_size=512, word_dim=1024, dropout_rate=0.3)
    network = model.Model(encoder=encoder, decoder=decoder) 

    start = time.time()
    for epoch in range(epochs_n):
        train(network, epoch + 1, train_loader=train_dataloader, loss_func=loss)

    end = time.time()
    torch.save(network, f"{MODEL_PATH}/model.pt")
    print(f"Training finished. Elapsed time: {end-start:.3f} seconds.")

if __name__ == "__main__":
    main()