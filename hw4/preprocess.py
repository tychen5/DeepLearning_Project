import numpy as np
import skimage
import skimage.io
import skimage.transform
import csv
import os
import random
import re

from vocab import Vocab


class Data(object):
    def __init__(self, img_id, img, tags, tag_text, with_text=1):
        self.img_id = img_id
        self.img = img
        self.tags = tags
        self.tag_text = tag_text
        self.with_text = with_text


# 對圖片進行quantization
def quantize(image, L=1, N=4):
    T = np.linspace(0, L, N, endpoint=False)[1:]
    return np.digitize(image.flat, T).reshape(image.shape)


class Sampler(object):
    def __init__(self, z_type):
        self.z_type = z_type

    def sample(self, batch_size, z_dim):
        return np.random.normal(0, 1, size=[batch_size, z_dim])


class DataManager(object):
    def __init__(self,
                 mode,
                 tag_file_path=None,
                 img_dir_path=None,
                 test_text_path=None,
                 vocab_path=None,
                 z_dim=100,
                 z_type='normal',
                 generator_output_layer='tanh'):

        self.index = {'train': 0, 'test': 0}

        self.tag_num = 2

        self.label_data = {}
        self.nega_data = {}

        self.z_sampler = Sampler(z_type=z_type)

        self.unk_counter = 0

        self.generator_output_layer = generator_output_layer

        if mode % 2 == 0:
            self.train_data, self.vocab = self.load_train_data(tag_file_path, img_dir_path)
            self.vocab.dump(vocab_path)
            self.find_negatives()
            self.gif_data = []
            for key in self.label_data:
                d = random.sample(self.label_data[key], 1)[0]
                self.gif_data.append(d)
            self.gif_z = self.z_sampler.sample(len(self.gif_data), z_dim)

        if mode // 2 == 0:
            # from cleaning import Vocab
            self.vocab = Vocab(vocab_path="./vocab")
            # self.vocab = Cleaning
            self.test_data = self.load_test_data(test_text_path, self.vocab)

    def load_train_data(self, tag_file_path, img_dir_path):

        vocab = Vocab(min_count=0)
        for color in ['blonde', 'aqua', 'white', 'orange', 'brown', 'black', 'blue', 'pink', 'purple', 'red',
                      'green', 'gray', 'yellow']:
            vocab.add_word(color)

        data = []
        # 資料前處理: 沒眼睛頭髮非顏色，TAG衝突者
        with open(tag_file_path, "r") as tag_f:
            reader = csv.reader(tag_f)
            for img_id, tag_str in reader:
                img_id = int(img_id)

                tags = [s.split(":")[0].strip() for s in tag_str.lower().split("\t")]
                hair = [t.split(" ")[0] for t in tags if t.endswith('hair')]
                eyes = [t.split(" ")[0] for t in tags if t.endswith('eyes')]

                hair = [vocab.encode(h) for h in hair if h in vocab.w2i and vocab.encode(h) != vocab.unknown]
                eyes = [vocab.encode(e) for e in eyes if e in vocab.w2i and vocab.encode(e) != vocab.unknown]
                if len(hair) == 0 and len(eyes) == 0:
                    continue

                if len(hair) > 1 or len(eyes) > 1:
                    continue
                with_text = 1
                with_unk = 0

                if len(hair) == 0 or len(hair) > 1 or len(eyes) == 0 or len(eyes) > 1:

                    if len(hair) == 1:
                        eyes = [vocab.encode(vocab.unknown)]
                        with_unk = 1

                    elif len(eyes) == 1:
                        hair = [vocab.encode(vocab.unknown)]
                        with_unk = 1

                    else:
                        hair = []
                        eyes = []
                        with_text = 0
                        with_unk = 1

                hair_str = [vocab.decode(h) for h in hair]
                eyes_str = [vocab.decode(e) for e in eyes]
                tag_text = "{}_hair_{}_eyes".format("_".join(hair_str), "_".join(eyes_str))

                hair = set(hair)
                eyes = set(eyes)
                feature = np.zeros((self.tag_num * vocab.vocab_size))

                for c_id in hair:
                    feature[c_id] += 1
                for c_id in eyes:
                    feature[c_id + vocab.vocab_size] += 1

                # image
                img_path = os.path.join(img_dir_path, str(img_id) + ".jpg")
                img = skimage.io.imread(img_path) / 127.5 - 1

                # 裁切大小
                img_resized = skimage.transform.resize(img, (64, 64), mode='constant')

                no_text = "{}_hair_{}_eyes".format('', '')

                if tag_text == no_text:
                    feature = np.zeros((self.tag_num * vocab.vocab_size)) / (vocab.vocab_size)

                # 增加資料量，進行旋轉平移
                for angle in [-20, -10, 0, 10, 20]:

                    img_rotated = skimage.transform.rotate(img_resized, angle, mode='edge')

                    for flip in [0, 1]:

                        if flip:
                            d = Data(img_id, np.fliplr(img_rotated), feature, tag_text, with_text)
                        else:
                            d = Data(img_id, img_rotated, feature, tag_text, with_text)

                        if tag_text not in self.label_data:
                            self.label_data[tag_text] = []

                        if with_text:
                            self.label_data[tag_text].append(d)

                        if with_unk:
                            self.unk_counter += 1

                        data.append(d)

        return data, vocab

    def parse_tag_text(self, tag_text):
        hair_str = re.findall('.*(?=_hair_)', tag_text)[0]
        eyes_str = re.findall('(?<=_hair_){1}.*(?=_eyes)', tag_text)[0]
        return hair_str, eyes_str

    def find_negatives(self):

        for tag_text1 in self.label_data:
            for tag_text2 in self.label_data:
                if tag_text1 != tag_text2:
                    hair_str1, eyes_str1 = self.parse_tag_text(tag_text1)
                    hair_str2, eyes_str2 = self.parse_tag_text(tag_text2)
                    if self.vocab.unknown in [hair_str1, hair_str2]:
                        if self.vocab.unknown not in [eyes_str1, eyes_str2] and eyes_str1 != eyes_str2:
                            if tag_text2 not in self.nega_data:
                                self.nega_data[tag_text2] = []
                            self.nega_data[tag_text2].extend(self.label_data[tag_text1])
                    elif self.vocab.unknown in [eyes_str1, eyes_str2]:
                        if self.vocab.unknown not in [hair_str1, hair_str2] and hair_str1 != hair_str2:
                            if tag_text2 not in self.nega_data:
                                self.nega_data[tag_text2] = []
                            self.nega_data[tag_text2].extend(self.label_data[tag_text1])
                    else:
                        if tag_text2 not in self.nega_data:
                            self.nega_data[tag_text2] = []
                        self.nega_data[tag_text2].extend(self.label_data[tag_text1])

    def load_test_data(self, test_text_path, vocab):
        data = []

        with open(test_text_path, "r") as f:
            reader = csv.reader(f)
            for text_id, text in reader:
                text_id = int(text_id)

                text_list = text.lower().split(" ")
                hair_color_id = vocab.encode(text_list[0])
                eyes_color_id = vocab.encode(text_list[2])

                feature = np.zeros((self.tag_num * vocab.vocab_size))

                feature[hair_color_id] += 1
                feature[eyes_color_id + vocab.vocab_size] += 1

                for img_id in range(1, 5 + 1):
                    data.append(Data("{}_{}".format(text_id, img_id), None, feature, text.lower().replace(" ", "_")))

        return data

    def draw_batch(self, batch_size, z_dim, mode='train'):
        if mode == 'train':
            data = self.train_data[self.index['train']: self.index['train'] + batch_size]
            if self.index['train'] + batch_size >= len(self.train_data):
                self.index['train'] = 0
                np.random.shuffle(self.train_data)
            else:
                self.index['train'] += batch_size
            noise = self.z_sampler.sample(len(data), z_dim)

            # noise
            noise_h = []
            wrong_img = []
            for d in data:
                nega_d = random.sample(self.nega_data[d.tag_text], 1)[0]
                noise_h.append(nega_d.tags)
                wrong_img.append(nega_d.img)

            return data, noise, noise_h, wrong_img

        if mode == 'test':
            data = self.test_data[self.index['test']: self.index['test'] + batch_size]
            if self.index['test'] + batch_size >= len(self.test_data):
                self.index['test'] = 0
            else:
                self.index['test'] += batch_size
            noise = self.z_sampler.sample(len(data), z_dim)
            return data, noise

        if mode == 'random':
            data = random.sample(self.train_data, batch_size)
            noise = self.z_sampler.sample(len(data), z_dim)
            return data, noise

        if mode == 'gif':
            data = self.gif_data
            noise = self.gif_z
            return data, noise

    def total_batch_num(self, batch_size, mode='train'):

        if mode == 'train':
            return int(np.ceil(len(self.train_data) / batch_size))

        if mode == 'test':
            return int(np.ceil(len(self.test_data) / batch_size))


if __name__ == "__main__":
    pass
