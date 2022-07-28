import os
import random

import numpy as np
from PIL import Image
from torchvision.transforms import Resize

from spirl.components.data_loader import VideoDataset as SpirlVideoDataset


def resize_video(video, size):
    if video.shape[1] == 3:
        video = np.transpose(video, (0,2,3,1))
    transformed_video = np.stack([np.asarray(Resize(size)(Image.fromarray(im))) for im in video], axis=0)
    return transformed_video


def shuffle_with_seed(arr, seed=0):
    rng = random.Random()
    rng.seed(seed)
    rng.shuffle(arr)
    return arr


def load_h5_files(dir):
    filenames = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".h5"): filenames.append(os.path.join(root, file))
    return filenames


class VideoDataset(SpirlVideoDataset):
    def __init__(self, *args, resolution, n_frames, **kwargs):
        super(VideoDataset, self).__init__(*args, resolution=resolution, **kwargs)
        self.n_frames = n_frames
        self.normalizing_max = None
        self.normalizing_min = None

    def __getitem__(self, index):
        data = self._get_raw_data(index)
        # print(data["images"].shape)
        # maybe subsample seqs
        if self.subsampler is not None:
            data = self._subsample_data(data)

        # sample random subsequence of fixed length
        if self.crop_subseq:
            end_ind = np.argmax(data.pad_mask * np.arange(data.pad_mask.shape[0], dtype=np.float32), 0)
            data = self._crop_rand_subseq(data, end_ind, length=self.spec.subseq_len)

        # Make length consistent
        start_ind = 0
        end_ind = np.argmax(data.pad_mask * np.arange(data.pad_mask.shape[0], dtype=np.float32), 0) \
            if self.randomize_length or self.crop_subseq else self.spec.max_seq_len - 1
        end_ind, data = self._sample_max_len_video(
            data, end_ind, target_len=self.spec.subseq_len if self.crop_subseq else self.spec.max_seq_len
        )

        if self.randomize_length:
            end_ind = self._randomize_length(start_ind, end_ind, data)
            data.start_ind, data.end_ind = start_ind, end_ind

        # perform final processing on data
        data.images = self._preprocess_images(data.images)      # [subseq_len, 32, 32, 3]
        data.last_images = data.images[-self.n_frames:]

        return data

    def _get_filenames(self):
        filenames = load_h5_files(self.data_dir)
        if not filenames:
            raise RuntimeError('No filenames found in {}'.format(self.data_dir))

        filenames = shuffle_with_seed(filenames)
        return filenames

    def _preprocess_images(self, images):
        # We don't divide the image by 255. It will be processed in network preprocessing.
        assert images.dtype == np.uint8
        images = resize_video(images, (self.img_sz, self.img_sz))
        return images
