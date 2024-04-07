import json
from PIL import Image
import os
from multiprocessing.pool import ThreadPool, Pool
import numpy as np
import itertools
import datasets
from time import time
import base64
import io
import torch
from torch.utils.data import Dataset
from datasets import DatasetDict, load_from_disk

torch.set_num_threads(1)

THREADS, CORES = os.cpu_count(), (os.cpu_count() - 4) // 2


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=base64str_to_PILobj(item["image"]), text=item["text"], padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        # import pdb; pdb.set_trace()
        encoding['label'] = item['label']
        return encoding


def imagefp_to_base64str(img_fp, format='JPEG'):
    img = Image.open(img_fp)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    # Encode the buffer content as Base64
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


def base64str_to_PILobj(base64_string):
    '''
    Args
    - base64_string (str): based64 encoded representing an image

    Output
    - PIL object (use .show() to display)
    '''
    image_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(image_data))
    # img.show()
    return img


def load_image_pil(t):
    t['image'] = imagefp_to_base64str(os.path.join('./hateful_memes/', t['img']), 'PNG')
    return t


def parallelize_cores(df, func, n_cores=CORES):
    df = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    out_lis = pool.map(func, df)
    pool.close()
    pool.join()
    return out_lis


def thr_load_img(t, t_cores=THREADS):
    t_pool = ThreadPool(t_cores)
    out = t_pool.starmap(load_image_pil, zip(t))
    t_pool.close()
    t_pool.join()
    return out


def tidy_and_save(file_type, try_run=False, save_dir='./processed_data_test/'):
    print('Reading file from ./hateful_memes/%s.jsonl'%(file_type))
    with open('./hateful_memes/%s.jsonl'%(file_type)) as f:
        if try_run:
            data = [json.loads(line) for line in f][:10]
        else:
            data = [json.loads(line) for line in f]

    data = parallelize_cores(data, thr_load_img)
    data = list(itertools.chain(*data))
    print(f'[{file}] Start dumpling data')

    s = time()
    ds = datasets.Dataset.from_list(data)
    if not os.path.exists(save_dir):
        print(f'Directory does not exist. Creating {save_dir}...')
        os.mkdir(save_dir)
    ds.save_to_disk(os.path.join(save_dir, "%s.hf"%(file_type)))
    print(f'File output to hf in {(time()-s):.2f}s')
    return


if __name__ == '__main__':
    OUT_PATH = './processed_data_test/'
    all_files = ['train', 'dev_seen', 'dev_unseen', 'test_seen', 'test_unseen']
    for file in all_files:
        tidy_and_save(file, try_run=True)

    combined_dict = {}
    for file in all_files:
        obj = load_from_disk(os.path.join(OUT_PATH, "%s.hf"%(file)))
        combined_dict[f'{file}'] = obj

    combined = DatasetDict(combined_dict)

    combined.save_to_disk(os.path.join(OUT_PATH, "combined_hateful_memes_dataset"))
    combined = load_from_disk(os.path.join(OUT_PATH, "combined_hateful_memes_dataset"))
    print(f'File successfully saved to {os.path.join(OUT_PATH, "combined_hateful_memes_dataset")}!')
