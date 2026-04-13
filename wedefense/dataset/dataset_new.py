import random
import torch
import torchaudio
import json
import soundfile as sf
from torch.utils.data import IterableDataset
import wedefense.dataset.shuffle_random_tools as shuffle_tools
import wedefense.dataset.processor as processor

class Processor(IterableDataset):
    def __init__(self, source, f, *args, **kw):
        assert callable(f)
        self.source = source
        self.f = f
        self.args = args
        self.kw = kw

    def set_epoch(self, epoch):
        if hasattr(self.source, 'set_epoch'):
            self.source.set_epoch(epoch)

    def __iter__(self):
        assert self.source is not None
        assert callable(self.f)
        return self.f(iter(self.source), *self.args, **self.kw)

    def apply(self, f):
        assert callable(f)
        return Processor(self, f, *self.args, **self.kw)

class DistributedSampler:
    def __init__(self, shuffle=True, partition=True, block_shuffle_size=0):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition
        self.block_shuffle_size = block_shuffle_size

    def update(self):
        self.rank = 0
        self.world_size = 1
        self.worker_id = 0
        self.num_workers = 1
        return dict(rank=self.rank, world_size=self.world_size, worker_id=self.worker_id, num_workers=self.num_workers)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def sample(self, data):
        """ Sample data according to rank/world_size/num_workers

            Args:
                data(List): input data list

            Returns:
                List: data list after sample
        """
        data = list(range(len(data)))
        if self.partition:
            if self.shuffle and self.block_shuffle_size > 0:
                # shuffle within blocks
                # e.g., [1,2,3,4,5,6], block_size=3 -> [3,1,2,5,4,6]
                shuffle_tools.f_shuffle_in_block_inplace(
                    data, self.block_shuffle_size, self.epoch)
                # shuffle blocks
                # e.g., [3,1,2,5,4,6], block_size=3 -> [5,4,6,3,1,2]
                shuffle_tools.f_shuffle_blocks_inplace(data,
                                                       self.block_shuffle_size,
                                                       self.epoch)

            elif self.shuffle:
                # default shuffle
                random.Random(self.epoch).shuffle(data)
            else:
                pass

            data = data[self.rank::self.world_size]
        data = data[self.worker_id::self.num_workers]
        return data

class DataList(IterableDataset):
    def __init__(self, lists, shuffle=True, partition=True, repeat_dataset=True, block_shuffle_size=0):
        self.lists = lists
        self.repeat_dataset = repeat_dataset
        self.sampler = DistributedSampler(shuffle, partition, block_shuffle_size)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        sampler_info = self.sampler.update()
        indexes = self.sampler.sample(self.lists)
        if not self.repeat_dataset:
            for index in indexes:
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data
        else:
            indexes_len = len(indexes)
            counter = 0
            while True:
                index = indexes[counter % indexes_len]
                counter += 1
                data = dict(src=self.lists[index])
                data.update(sampler_info)
                yield data

def parse_multi_raw(iter_src, target_dur = 5.0):
    """Parse sample with multiple wav paths into list of tensors"""
    for data in iter_src:
        if isinstance(data['src'], str):
            data_dict = json.loads(data['src'])
        else:
            data_dict = data['src']

        wav_paths = data_dict.get('wav') or data_dict.get('wavs')
        if not isinstance(wav_paths, list):
            wav_paths = [wav_paths]

        wav_tensors = []
        sr = None
        for path in wav_paths:
            try:
                wav, sr_ = torchaudio.load(path, normalize=True)
                wav = wav.mean(dim=0)
            except:
                wav_np, sr_ = sf.read(path)
                wav = torch.tensor(wav_np, dtype=torch.float)
                if wav.ndim > 1:
                    wav = wav.mean(dim=1)
            sr = sr_
            target_len = int(sr * target_dur)
            if wav.size(0) > target_len:
                start = random.randint(0, wav.size(0) - target_len)
                wav = wav[start:start + target_len]
            elif wav.size(0) < target_len:
                pad_len = target_len - wav.size(0)
                wav = torch.nn.functional.pad(wav, (0, pad_len))

            wav_tensors.append(wav)

        data_dict['wav'] = wav_tensors
        data_dict['sr'] = sr
        yield data_dict

def multi_compute_fbank(data, **fbank_args):
    """
    data: Iterable[sample_dict]
      sample_dict['wav'] is List[Tensor], each Tensor shape (1, T)
    output:
      sample_dict['feat'] becomes List[Tensor], each Tensor shape (Frames, Mel) or (Mel, Frames) depending on your compute_fbank
      and sample_dict['wav'] can be deleted to save memory (optional)
    """
    from wedefense.dataset import processor

    for sample in data:
        wavs = sample.get("wav", None)
        if not isinstance(wavs, list) or len(wavs) == 0:
            yield sample
            continue

        sr = int(sample.get("sample_rate", sample.get("sr", 16000)))
        feats = []

        for i, w in enumerate(wavs):
            # ensure (1, T)
            if w.dim() == 1:
                w = w.unsqueeze(0)

            one = {
                "key": f"{sample.get('key', 'utt')}|w{i}",
                "wav": w,
                "sample_rate": sr,
            }
            if "label" in sample:
                one["label"] = sample["label"]

            out = next(processor.compute_fbank([one], **fbank_args))

            if "feat" in out:
                feats.append(out["feat"])
            elif "fbank" in out:
                feats.append(out["fbank"])
            else:
                raise KeyError("compute_fbank output has no 'feat'/'fbank' field.")

        sample["feat"] = feats
        yield sample


def Dataset(data_type, data_list_file, configs, lab2id_dict, whole_utt=False, reverb_lmdb_file=None, noise_lmdb_file=None, repeat_dataset=True, data_dur_file=None, block_shuffle_size=0, target_dur = 5.0):
    from wedefense.utils.file_utils import read_lists
    lists = read_lists(data_list_file)

    dataset = DataList(lists, shuffle=configs.get('shuffle', False), repeat_dataset=repeat_dataset, block_shuffle_size=block_shuffle_size)
    if data_type == 'shard':
        from wedefense.dataset import processor
        dataset = Processor(dataset, processor.url_opener)
        dataset = Processor(dataset, processor.tar_file_and_group)
    elif data_type == 'raw':
        dataset = Processor(dataset, parse_multi_raw, target_dur=target_dur)
    else:
        from wedefense.dataset import processor
        dataset = Processor(dataset, processor.parse_feat)

    from wedefense.dataset import processor
    dataset = Processor(dataset, processor.lab_to_id, lab2id_dict)
    return dataset