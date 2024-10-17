import sys
import json
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import train
import bleu_eval

def load_model(model_path):
    return torch.load(model_path)

def process_test_data(test_data_path, model, index_to_word):
    test_dataset = train.TestDataLoader(test_data_path)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, num_workers=8)
    return train.test(test_dataloader, model, index_to_word)

def write_output(output_path, results):
    try:
        with open(output_path, 'w') as f:
            for id, s in results:
                f.write(f'{id},{s}\n')
        print('File updated successfully!')
    except FileNotFoundError:
        with open(output_path, 'x') as f:
            for id, s in results:
                f.write(f'{id},{s}\n')
        print('File created and updated successfully!')

def load_test_json(test_json_path):
    with open(test_json_path, 'r') as f:
        return json.load(f)

def read_output_file(output_path):
    result = {}
    with open(output_path, 'r') as f:
        for line in f:
            test_id, caption = line.rstrip().split(',', 1)
            result[test_id] = caption
    return result

def calculate_bleu_scores(test_data, result):
    bleu_scores = []
    for item in test_data:
        captions = [x.rstrip('.') for x in item['caption']]
        bleu_scores.append(bleu_eval.BLEU(result[item['id']], captions, True))
    return bleu_scores

def main():
    test_data_path = sys.argv[1]
    test_json_path = "MLDS_hw2_1_data/testing_label.json"
    model_path = "Model/model.pt"
    output_path = sys.argv[2]

    model = load_model(model_path)
    index_to_word = train.create_dictionary(4)

    results = process_test_data(test_data_path, model, index_to_word)
    write_output(output_path, results)

    test_data = load_test_json(test_json_path)
    output_result = read_output_file(output_path)
    bleu_scores = calculate_bleu_scores(test_data, output_result)

    average_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score is {average_bleu:.4f}")

if __name__ == '__main__':
    mp.freeze_support()
    main()