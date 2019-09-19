import csv
import re
from random import seed, sample, shuffle
import requests

from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

from visuals import plot_history

def fetch_puns_list(num_pages):
    base_url = 'https://onelinefun.com/puns/{page}/'
    puns = []
    for page in range(1, num_pages+1):
        url = base_url.format(page=page)
        request = requests.get(url)
        if request.status_code != 200:
            raise RuntimeError(f'Attempt to request page #{page} returned an invalid response code: {request.status_code}')
        else:
            page_puns = []
            soup = BeautifulSoup(request.content, 'html.parser')
            for div in soup.findAll('div', {'class': 'o'}):
                regex = re.compile(r"""([\"'])(?:(?=(\\?))\2.)*?\1""")
                div_puns = [[pun.getText(), 1] if not re.match(regex, pun.getText()) else None for pun in div.findAll('p')]
                page_puns.extend(div_puns)
            puns.extend(page_puns)
    puns = [item for item in puns if item]
    return puns

def add_padding(data):
    n_sampels = len(data)
    max_seq_length = max(map(len, data))

    data_matrix = np.zeros((n_sampels, max_seq_length))
    for i, sample in enumerate(data):
        data_matrix[i, -len(sample):] = sample

    return data_matrix

def make_sentence_examples(path='/data/movie_lines.tsv'):
    with open(path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        data = [[row[-1], 0] for row in reader]
        seed(614)
        return sample(data, 990)

def make_train_csv(out_file='/data/training.csv', num_pages=99):
    sentences = make_sentence_examples()
    puns = fetch_puns_list(num_pages)
    output = sentences + puns 
    shuffle(output)
    with open(out_file, 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(output)

def read_dataset(dataset='/data/training.csv', on_error='generate', num_pages=99):
    try:
        with open(dataset, 'r') as infile:
            reader = csv.reader(infile, delimiter=',', escapechar='\\')
            data = [[row[0], int(row[1])] for row in reader]
            return np.array(data)
    except FileNotFoundError:
        if on_error == 'generate':
            make_train_csv(out_file=dataset, num_pages=num_pages)
            return read_dataset(dataset=dataset, on_error='raise')
        elif on_error == 'raise':
            raise RuntimeError
    


def make_train_test(dataset='/data/training.csv', ratio=0.40, default_error_behavior='generate', maxlen=256):
    data = read_dataset(dataset=dataset, on_error=default_error_behavior)
    examples = data[:, :-1]
    labels = data[:, -1:]
    train_examples, test_examples, train_labels, test_labels = train_test_split(examples, labels, test_size=ratio)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(examples.ravel().tolist())
    

    train_examples = tokenizer.texts_to_sequences(train_examples.ravel().tolist())
    test_examples = tokenizer.texts_to_sequences(test_examples.ravel().tolist())

    vocab_size = len(tokenizer.word_index) + 1

    train_examples = pad_sequences(train_examples, padding='post', maxlen=maxlen)
    test_examples= pad_sequences(test_examples, padding='post', maxlen=maxlen)

    return train_examples, test_examples, train_labels, test_labels, vocab_size    

if __name__ == '__main__':
    print(make_sentence_examples())
