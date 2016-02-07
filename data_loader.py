# -*- coding: utf-8 -*-
import os

import numpy as np
from scipy.io.wavfile import read


class DataLoader(object):
    def __init__(self, config):
        self.data_dir = './data/timit/'

        self.batch_size = config.batch_size
        self.num_steps = config.num_steps

        self.data = self.load_data()
        self.num_batches = len(self.data) / self.batch_size
        np.random.shuffle(self.data)

    def load_data(self):
        data = []
        dirs = [f for f in os.listdir(self.data_dir) if f.startswith('dr')]

        for dir in dirs:
            dd = np.array(self.load_wav_set(dir))
            for d in dd:
                for x in d:
                    if 'dr2-faem0' in dir:
                        y = np.ones(x.shape)
                    else:
                        y = np.zeros(x.shape)
                    data.append([x, y])

        return np.array(data)

    def load_wav_set(self, path):
        wav_files = [f for f in os.listdir(os.path.join(self.data_dir, path)) if f.endswith('.wav')]

        data = []
        for f in wav_files:
            d = self.load_wav(os.path.join(path, f))
            c = self.chunk_data(d)
            data.append(c)

        return np.array(data).flatten()

    def load_wav(self, path):
        w = read(os.path.join(self.data_dir, path))
        data = np.array(w[1], dtype=float)

        d = []
        for x in data:
            d.append([x])

        return d

    def chunk_data(self, data):
        chunked = []
        i = 0
        while i + self.num_steps < len(data):
            d = data[i:i + self.num_steps]
            i += self.num_steps
            chunked.append(d)
        return np.array(chunked)

    def next_batch(self):
        x_batch = []
        y_batch = []

        np.random.shuffle(self.data)

        data = []
        for i in xrange(self.batch_size):
            data.append(self.data[i])

        for d in data:
            x_batch.append(d[0])
            y_batch.append(d[1])

        return x_batch, y_batch
