import logging
import os
import random
from os.path import join

import cv2
import numpy as np
import torch
from decord import VideoReader
from decord import cpu
from pytorch_lightning import LightningDataModule
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm import tqdm

from config import *


def subsample_list(inp_list: list, sample_rate: float):
    random.shuffle(inp_list)
    return [inp_list[i] for i in range(int(len(inp_list) * sample_rate))]


class AVSEDataset(Dataset):
    def __init__(self, scenes_root, shuffle=True, seed=SEED, subsample=1,
                 clipped_batch=True, sample_items=True, test_set=False, lips=False, rgb=False):
        super(AVSEDataset, self).__init__()
        if lips:
            self.img_width, self.img_height = 96, 96
        else:
            self.img_width, self.img_height = 224, 224
        self.lips = lips
        self.test_set = test_set
        self.clipped_batch = clipped_batch
        self.scenes_root = scenes_root
        self.files_list = self.build_files_list
        if shuffle:
            random.seed(SEED)
            random.shuffle(self.files_list)
        if subsample != 1:
            self.files_list = subsample_list(self.files_list, sample_rate=subsample)
        logging.info("Found {} utterances".format(len(self.files_list)))
        self.data_count = len(self.files_list)
        self.batch_index = 0
        self.total_batches_seen = 0
        self.batch_input = {"noisy": None}
        self.index = 0
        self.max_len = len(self.files_list)
        self.max_cache = 0
        self.seed = seed
        self.window = "hann"
        self.fading = False
        self.rgb = rgb
        self.sample_items = sample_items

    @property
    def build_files_list(self):
        files_list = []
        for file in os.listdir(self.scenes_root):
            if file.endswith("mixed.wav"):
                if self.lips:
                    files = (join(self.scenes_root, file.replace("mixed", "target")),
                             join(self.scenes_root, file.replace("mixed", "interferer")),
                             join(self.scenes_root, file),
                             join(self.scenes_root.replace("scenes", "lips"), file.replace("_mixed.wav", "_silent.mp4")),
                             )
                else:
                    files = (join(self.scenes_root, file.replace("mixed", "target")),
                             join(self.scenes_root, file.replace("mixed", "interferer")),
                             join(self.scenes_root, file),
                             join(self.scenes_root, file.replace("_mixed.wav", "_silent.mp4")),
                             )
                if not self.test_set:
                    if all([isfile(f) for f in files]):
                        files_list.append(files)
                else:
                    files_list.append(files)
        return files_list

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        while True:
            try:
                data = {}
                if self.sample_items:
                    clean_file, noise_file, noisy_file, mp4_file = random.sample(self.files_list, 1)[0]
                else:
                    clean_file, noise_file, noisy_file, mp4_file = self.files_list[idx]
                data["noisy_audio"], data["clean"], data["video_frames"] = self.get_data(clean_file, noise_file,
                                                                                         noisy_file, mp4_file)
                data['scene'] = clean_file.replace(self.scenes_root, "").replace("_target.wav", "").replace("/", "")
                return data
            except Exception as e:
                logging.error("Error in loading data: {}".format(e))

    def load_wav(self, wav_path):
        return wavfile.read(wav_path)[1].astype(np.float32) / (2 ** 15)

    def get_data(self, clean_file, noise_file, noisy_file, mp4_file):
        noisy = self.load_wav(noisy_file)
        vr = VideoReader(mp4_file, ctx=cpu(0))
        if isfile(clean_file):
            clean = self.load_wav(clean_file)
        else:
            # clean file for test set is not available
            clean = np.zeros(noisy.shape)
        if self.clipped_batch:
            if clean.shape[0] > max_audio_len:
                clip_idx = random.randint(0, clean.shape[0] - max_audio_len)
                video_idx = int((clip_idx / sampling_rate) * frames_per_second)
                clean = clean[clip_idx:clip_idx + max_audio_len]
                noisy = noisy[clip_idx:clip_idx + max_audio_len]
            else:
                video_idx = -1
                clean = np.pad(clean, pad_width=[0, max_audio_len - clean.shape[0]], mode="constant")
                noisy = np.pad(noisy, pad_width=[0, max_audio_len - noisy.shape[0]], mode="constant")
            if len(vr) < max_frames:
                frames = vr.get_batch(list(range(len(vr)))).asnumpy()
            else:
                max_idx = min(video_idx + max_frames, len(vr))
                frames = vr.get_batch(list(range(video_idx, max_idx))).asnumpy()
            if not self.rgb:
                bg_frames = np.array(
                    [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(len(frames))]).astype(np.float32)
            else:
                bg_frames = np.array(frames).astype(np.float32)
            bg_frames /= 255.0
            if len(bg_frames) < max_frames:
                if not self.rgb:
                    bg_frames = np.concatenate(
                        (bg_frames, np.zeros((max_frames - len(bg_frames), self.img_height, self.img_width)).astype(bg_frames.dtype)),
                        axis=0)
                else:
                    bg_frames = np.concatenate(
                        (bg_frames, np.zeros((max_frames - len(bg_frames), self.img_height, self.img_width, 3)).astype(bg_frames.dtype)),
                        axis=0)
        else:
            frames = vr.get_batch(list(range(len(vr)))).asnumpy()
            if not self.rgb:
                bg_frames = np.array(
                    [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in range(len(frames))]).astype(np.float32)
            else:
                bg_frames = np.array(frames).astype(np.float32)
            bg_frames /= 255.0
        if not self.rgb:
            return noisy, clean, bg_frames[np.newaxis, ...]
        else:
            return noisy, clean, bg_frames.transpose(0, 3, 1, 2)


class AVSEDataModule(LightningDataModule):
    def __init__(self, batch_size=16, lips=False):
        super(AVSEDataModule, self).__init__()
        self.train_dataset_batch = AVSEDataset(join(DATA_ROOT, "train/scenes"), lips=lips)
        self.dev_dataset_batch = AVSEDataset(join(DATA_ROOT, "dev/scenes"), lips=lips)
        self.dev_dataset = AVSEDataset(join(DATA_ROOT, "dev/scenes"), clipped_batch=False,
                                       sample_items=False, lips=lips)
        self.eval_dataset = AVSEDataset(join(DATA_ROOT, "eval/scenes"), clipped_batch=False,
                                        sample_items=False, lips=lips, test_set=True)
        self.batch_size = batch_size

    def train_dataloader(self):
        assert len(self.train_dataset_batch) > 0, "No training data found"
        return torch.utils.data.DataLoader(self.train_dataset_batch, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        assert len(self.dev_dataset_batch) > 0, "No validation data found"
        return torch.utils.data.DataLoader(self.dev_dataset_batch, batch_size=self.batch_size, num_workers=4,
                                           pin_memory=True,
                                           persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dev_dataset, batch_size=self.batch_size, num_workers=4)


if __name__ == '__main__':

    dataset = AVSEDataModule(batch_size=1, lips=False).train_dataset_batch
    for i in tqdm(range(len(dataset)), ascii=True):
        data = dataset[i]
        for k, v in data.items():
            if type(v) == np.ndarray:
                print(k, v.shape, "Max:-", v.max(), "Min:-", v.min())
        break