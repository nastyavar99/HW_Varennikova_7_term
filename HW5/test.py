import re
import random

from sklearn.datasets import fetch_20newsgroups

import numpy as np
from scipy.spatial.distance import cdist

import nltk
# nltk.download() <-- сюда впишите строкой то, чего ему не хватит при работе базовых функций

from nltk import wordpunct_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer

import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import RegexpTokenizer

import tqdm

cats = ['alt.atheism', 'sci.space', 'rec.sport.hockey', 'sci.med']
data = fetch_20newsgroups(subset='train', categories=cats)['data']

np.random.shuffle(data)

stemmer = PorterStemmer()

tokenizer = RegexpTokenizer(r'\w+')


def preprocess(text):
    # split_regex = re.compile(r'[.|!|?|…]')
    templates = {
        'From:', 'Subject:', 'Organization:', 'Lines:', 'NNTP-Posting-Host:', 'Distribution:', 'Keywords:',
        'News-Software', ') writes:', 'In-Reply-To:', 'X-News-Reader:', 'Nntp-Posting-Host:', '> writes:', ' writes:',
        '.edu', ') wrote:', 'Article-I.D.:', 'In article', 'X-Newsreader:', '[', ']', '.com', 'Reply-To:', ') says:',
        'X-Mailer:', 'Summary:', ') writes:'
    }
    splitted_text = text.split('\n')
    text_necessary = []
    for line in splitted_text:
        if line:
            text_necessary.append(line)
            for t in templates:
                if t in line:
                    text_necessary.remove(line)
                    break

    cleared_text = ' '.join(text_necessary)
    # sents = re.split(r'(?<=[.!?…]) ', cleared_text)
    sents = sent_tokenize(cleared_text)

    result = []
    for s in sents:
        no_tabs = s.replace('\t', ' ')
        no_multi_spaces = re.sub(" +", " ", no_tabs)
        no_spaces = no_multi_spaces.strip()

        if '@' in s:
            continue

        words = wordpunct_tokenize(no_spaces)
        words_lst = []
        for w in words:
            if w.isalpha():
                words_lst.append(stemmer.stem(w))
        result.append(words_lst)
    return result


data = [preprocess(text) for text in data[:100]]
data = [item for sublist in data for item in sublist]
data = [lst for lst in data if lst]

vocabulary = set()
for sent in data:
    vocabulary = vocabulary.union({w for w in sent})

word2idx = {}
for i, w in enumerate(vocabulary):
    word2idx[w] = i

idx2word = {}
for k, v in word2idx.items():
    idx2word[v] = k


def make_contexts(sent, window=2):
    # INPUT: токенизированное предложение
    # OUTPUT: вложенный список контекстов.
    # Каждый контекст состоит из двух токенов: центральный и сосед
    contexts = []
    for i in range(len(sent)):
        central_word = sent[i]
        if i < window:
            new_left_window = window - i
            left_context = sent[i - new_left_window:i]
        else:
            left_context = sent[i - window:i]
        right_context = sent[i + 1:i + window + 1]
        context = [[central_word, neighbour] for neighbour in left_context + right_context]
        contexts.extend(context)
    return contexts


# Делаем контексты
data = [make_contexts(sent) for sent in data]

# Опять flattener ...
data = [item for sublist in data for item in sublist]

# Переделали кортежи слов в кортежи индексов
idx_pairs = [(word2idx[center], word2idx[context]) for center, context in data]


def create_batches(pairs_of_idx, batch_size):
    random.shuffle(pairs_of_idx)
    return np.array_split(np.array(pairs_of_idx), batch_size)


BATCHES = create_batches(idx_pairs, 100)


# Теперь реализуем архитектуру Word2Vec.
# Word2vec - это нейросеть с одним скрытым слоем.
# На лекции мы говорили, что для того, чтобы выбрать нужный вектор столбец из нашего "первого слоя"
# (хотя к нему можно относиться и как к входному) нужно умножить one-hot вектор на соответствующую матрицу
# размера (vocab_size, emb_size).
# В pytorch всё можно сделать ещё проще. Пропустим шаг с созданием one-hot вектора и сразу же
# извлечем наш вектор из слоя, который называется nn.Embedding с помощью индекса нашего токена (помните  word2idx?)
# Затем (без активации) пропустим через наш простой линейный (i.e. полносвязный) слой
# После активации мы получаем распределение вероятностей для предсказанного токена.

class W2V(nn.Module):
    def __init__(self, vocab_size, word2idx, idx2word, emb_size=5):
        super(W2V, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.word2idx = word2idx
        self.idx2word = idx2word

        # Матрица u --> эмбеддинг слой, который мы ожидаем, что будет содержать в себе семантику слов после обучения
        # Размер (vocab_size, emb_size). Задайте ему также параметр sparse
        self.u_emb = nn.Embedding(self.vocab_size, self.emb_size, sparse=True)

        # Матрица v --> преобразует скрытое состояние в вектор (или матрицу, если батч) размера количества слов
        # Размер (emb_size, vocab_size)
        self.v_emb = nn.Linear(self.emb_size, self.vocab_size)

    def __getitem__(self, word):
        """
            get embedding of a word of shape (1, emb_size)
        """
        return self.u_emb(torch.tensor(self.word2idx[word]))

    def top_k_closest(self, word, k):
        """
            top-k cosine-closest words
        """
        idx = self.word2idx[word]
        word_tensor = self.u_emb(torch.tensor([idx]))

        cos = nn.CosineSimilarity()
        distances_between_embeddings = {}
        for w, v in self.word2idx.items():
            output = cos(word_tensor, self.u_emb(torch.tensor([v])))
            distances_between_embeddings[w] = output.item()

        sorted_words_list = sorted(distances_between_embeddings, key=distances_between_embeddings.get, reverse=True)
        return sorted_words_list[1:k + 1]

    def forward(self, center_word_idx):
        """
           x: idx of center word
        """
        center_word_tensor = self.u_emb(center_word_idx)  # forward emb layer
        vector = self.v_emb(center_word_tensor)  # forward emb layer
        return vector


emb_size = 50
num_epochs = 200
learning_rate = 0.001

model = W2V(len(idx2word), word2idx, idx2word)

# Cross Entropy
criterion = nn.CrossEntropyLoss()

# SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

softmax = nn.Softmax()

# Напишем наш цикл обучения. Попробуем реализовать простенько, без батчей.
# Но будем копить градиент и каждые 100 итераций обновлять веса

for epo in range(num_epochs):

    loss_val = 0
    # iterations = 1
    # for batch in tqdm.tqdm(BATCHES):
    for i, (center, context) in tqdm.tqdm(enumerate(idx_pairs)):
        center = torch.tensor(center, dtype=torch.long).unsqueeze(0)
        context = torch.tensor(context, dtype=torch.long).unsqueeze(0)
        # print(center)
        # Пропускаем слово через модель
        z = model(center)
        # print(z)
        # Активируем софтмаксом
        distribution = softmax(z)

        # Считаем лосс (он принимает распределение предсказанных вероятностей и правильный ответ)
        # Скорее всего нужно сделать reshape с помощью view
        loss = criterion(distribution, context)

        # Копим общий лосс, и надеямся, что на следующей эпохи накопится поменьше
        loss_val += loss

        # Проверяем, прошло ли 100 итераций.
        if i == 100:
            optimizer.zero_grad()  # 1) зануляем градиенты с предыдущего шага оптимизатора
            loss.backward()  # 2) backprop
            optimizer.step()  # 3) шаг оптимизации

        # iterations += 1

    # Выведем усредненный на количество пар лосс
    print(f'Loss at epo {epo}: {loss_val / len(idx_pairs)}')

x = 10
