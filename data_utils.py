import torch
from torch.utils.data import Dataset, DataLoader
from nnmnkwii.datasets import vctk
import math
import random
import numpy as np
import os

from text import text_to_sequence
import Audio
import hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tacotron2Dataset(Dataset):
    """ VCTK """

    def __init__(self):
        speakers = list()
        for file in os.listdir(os.path.join(hparams.VCTK_dataset_path, "wav48")):
            speakers.append(str(file[1:4]))
        self.speaker_list = speakers

        # Shuffle Speakers
        if not os.path.exists(hparams.shuffle_speaker_file):
            shuffle_list = [i for i in range(len(self.speaker_list))]
            random.shuffle(shuffle_list)
            shuffle_list = shuffle_list[0 : hparams.shuffle_speaker_num]
            self.speaker_list = self.shuffle(self.speaker_list, shuffle_list)

            with open(hparams.shuffle_speaker_file, "w") as f:
                for speaker in shuffle_list:
                    f.write(str(speaker) + "\n")
        else:
            shuffle_list = list()
            with open(hparams.shuffle_speaker_file, "r") as f:
                for ele in f.readlines():
                    shuffle_list.append(int(ele))

            self.speaker_list = self.shuffle(self.speaker_list, shuffle_list)

        td = vctk.TranscriptionDataSource(
            hparams.VCTK_dataset_path, speakers=self.speaker_list
        )
        transcriptions = td.collect_files()
        wav_paths = vctk.WavFileDataSource(
            hparams.VCTK_dataset_path, speakers=self.speaker_list
        ).collect_files()

        self.text = transcriptions
        self.wav_paths = wav_paths

    def shuffle(self, list_speaker, shuffle_list):
        out = list()
        for ind, ele in enumerate(list_speaker):
            if ind not in shuffle_list:
                out.append(ele)

        return out

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        speaker_id = self.speaker_list.index(self.wav_paths[idx][36:39])
        mel_target = Audio.tools.get_mel(self.wav_paths[idx]).numpy().T

        character = self.text[idx]
        character = text_to_sequence(character, hparams.text_cleaners)
        character = np.array(character)

        stop_token = np.array([0.0 for _ in range(mel_target.shape[0])])
        stop_token[-1] = 1.0

        sample = {
            "text": character,
            "mel_target": mel_target,
            "stop_token": stop_token,
            "speaker_id": speaker_id,
        }

        return sample


def _process(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    stop_tokens = [batch[ind]["stop_token"] for ind in cut_list]
    indexs = [batch[ind]["speaker_id"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.shape[0])

    length_mel = np.array([])
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    texts = pad_normal(texts)
    stop_tokens = pad_normal(stop_tokens, PAD=1.0)
    mel_targets = pad_mel(mel_targets)

    out = {
        "text": texts,
        "mel_target": mel_targets,
        "stop_token": stop_tokens,
        "length_mel": length_mel,
        "length_text": length_text,
        "speaker_id": indexs,
    }

    return out


def collate_fn(batch):
    len_arr = np.array([d["text"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i * real_batchsize : (i + 1) * real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(_process(batch, cut_list[i]))

    return output


def pad_normal(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
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
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
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
        input_lengths = torch.from_numpy(batchs[0]["length_text"]).int().to(device)
        mel_target = (
            torch.from_numpy(batchs[0]["mel_target"])
            .float()
            .to(device)
            .contiguous()
            .transpose(1, 2)
        )
        stop_target = torch.from_numpy(batchs[0]["stop_token"]).float().to(device)
        output_lengths = torch.from_numpy(batchs[0]["length_mel"]).int().to(device)

        speaker_id = torch.Tensor([0 for _ in range(hparams.speaker_size)])
        speaker_id[0] = 1
        speaker_id = speaker_id.float().to(device)
        batch = (
            character,
            input_lengths,
            mel_target,
            stop_target,
            output_lengths,
            speaker_id,
        )

        out_dataset.append(batch)

    out = list()
    for ind in sample_list:
        out.append(out_dataset[ind])
    print("Got Old Dataset.")

    return out


if __name__ == "__main__":
    # Test
    test_dataset = Tacotron2Dataset()
