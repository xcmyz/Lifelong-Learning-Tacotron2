import torch
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import random
import os

from text import text_to_sequence
import hparams

device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')


class Tacotron2Dataset(Dataset):
    """ LJSpeech """

    def __init__(self, dataset_path=hparams.dataset_path):
        self.dataset_path = dataset_path
        self.text_path = os.path.join(self.dataset_path, "train.txt")
        self.text = process_text(self.text_path)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        index = idx + 1
        mel_name = os.path.join(
            self.dataset_path, "ljspeech-mel-%05d.npy" % index)
        mel_target = np.load(mel_name)

        character = self.text[idx][0:len(self.text[idx])-1]
        character = text_to_sequence(character, hparams.text_cleaners)
        character = np.array(character)

        stop_token = np.array([0. for _ in range(mel_target.shape[0])])
        stop_token[-1] = 1.

        sample = {"text": character, "mel_target": mel_target,
                  "stop_token": stop_token}

        return sample


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def _process(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    stop_tokens = [batch[ind]["stop_token"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.shape[0])

    length_mel = np.array([])
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    texts = pad_normal(texts)
    stop_tokens = pad_normal(stop_tokens, PAD=1.)
    mel_targets = pad_mel(mel_targets)

    out = {"text": texts,  "mel_target": mel_targets, "stop_token": stop_tokens,
           "length_mel": length_mel, "length_text": length_text}

    return out


def collate_fn(batch):
    len_arr = np.array([d["text"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(_process(batch, cut_list[i]))

    return output


def pad_normal(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode='constant', constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_mel(inputs):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)
                              [0]), mode='constant', constant_values=PAD)
        return x_padded[:, :s]

    max_len = max(np.shape(x)[0] for x in inputs)
    mel_output = np.stack([pad(x, max_len) for x in inputs])

    return mel_output


def get_old_dataset(dataset_path, sample_num):
    out_dataset = list()
    dataset = Tacotron2Dataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    len_dataloader = len(dataloader)
    sample_list = [i for i in range(len_dataloader)]
    random.shuffle(sample_list)
    sample_list = sample_list[:sample_num]

    for _, batchs in enumerate(dataloader):
        character = torch.from_numpy(batchs[0]["text"]).long().to(device)
        input_lengths = torch.from_numpy(
            batchs[0]["length_text"]).int().to(device)
        mel_target = torch.from_numpy(batchs[0]["mel_target"]).float().to(
            device).contiguous().transpose(1, 2)
        stop_target = torch.from_numpy(
            batchs[0]["stop_token"]).float().to(device)
        output_lengths = torch.from_numpy(
            batchs[0]["length_mel"]).int().to(device)

        speaker_id = torch.Tensor([0 for _ in range(hparams.speaker_size)])
        speaker_id[0] = 1
        speaker_id = speaker_id.float().to(device)
        batch = character, input_lengths, mel_target, stop_target, output_lengths, speaker_id

        out_dataset.append(batch)

    out = list()
    for ind in sample_list:
        out.append(out_dataset[ind])
    print("Got Old Dataset.")

    return out
