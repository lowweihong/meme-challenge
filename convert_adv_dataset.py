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

THREADS, CORES = os.cpu_count(), os.cpu_count()

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


def load_image_pil(t,img_col='img'):
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


def tidy_and_save(file_type, try_run=False, save_dir='./processed_adv_data/'):
    print('Reading file from ./adversarial_robustness/adv_datasets/%s.jsonl'%(file_type))
    with open('./adversarial_robustness/adv_datasets/%s.jsonl'%(file_type)) as f:
        if try_run:
            data = [json.loads(line) for line in f][:10]
        else:
            data = [json.loads(line) for line in f]

    data = parallelize_cores(data, thr_load_img)
    data = list(itertools.chain(*data))
    print(f'[{file}] Start dumping data')
    s = time()
    ds = datasets.Dataset.from_list(data)
    if not os.path.exists(save_dir):
        print(f'Directory does not exist. Creating {save_dir}...')
        os.mkdir(save_dir)
    ds.save_to_disk(os.path.join(save_dir, "%s.hf"%(file_type)))
    print(f'File output to hf in {(time()-s):.2f}s')
    return


if __name__ == '__main__':
    OUT_PATH = './processed_adv_data/'

    all_files = ['adv_dev_unseen',
                'adv_test_unseen',
                'emoji_aug_dev_unseen',
                'grayscale_aug_dev_unseen',
                'added_noise_aug_dev_unseen',
                'simulate_typos_dev_unseen',
                'swap_gendered_words_dev_unseen',
                'similar_unicode_dev_unseen',
                'emoji_aug_test_unseen',
                'grayscale_aug_test_unseen',
                'added_noise_aug_test_unseen',
                'simulate_typos_test_unseen',
                'swap_gendered_words_test_unseen',
                'similar_unicode_test_unseen',
                'augmented_train_w_txt',
                'combined_dev_unseen',
                'combined_test_unseen',
                'dev_seen'
                ]

    #for file in all_files:
    #    tidy_and_save(file, try_run=False, save_dir = OUT_PATH)


    combined_dict = {}
    for file in all_files:
        obj = load_from_disk(os.path.join(OUT_PATH, "%s.hf"%(file)))
        combined_dict[f'{file}'] = obj

    combined = DatasetDict(combined_dict)

    out_filepath = os.path.join(OUT_PATH, "combined_adversarial_dataset")

    combined.save_to_disk(out_filepath)
    print(f'File successfully saved to {out_filepath}!')

