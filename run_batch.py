import threading

import numpy as np
import os
import torch
import argparse

base_cmd = 'python -u evaluate_topic.py ' \
           '--ckpt lm-prediction/ckpt/topic/future_word_predictor/model.pth.tar ' \
           '--dataset_info lm-prediction/ckpt/topic/future_word_predictor/dataset_info ' \
           '--prefix_file PREFIX ' \
           '--wordlist_dir topic_data/wordlists_gender/ ' \
           '--condition_lambda 0.0 1.0 2.0 4.0 8.0 12.0 ' \
           '--verbose ' \
           '--precondition_topk 200 ' \
           '--topk 10 ' \
           '--sample_size NUM_SAMPLES ' \
           '--max_sample_batch 1 ' \
           '--length_cutoff 15 ' \
           '--log_file OUT_DIR ' \
           '--device DEVICE'


def run(device_num: int, num_samples: int, out_dir: str):
    cmd = base_cmd \
        .replace('NUM_SAMPLES', f'{num_samples:d}') \
        .replace('DEVICE', f'cuda:{device_num}') \
        .replace('PREFIX', f'topic_data/tmp/topics_{device_num}.txt')\
        .replace('OUT_DIR', out_dir)
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument('--contexts', type=str, required=True)
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--out-dir', type=str, required=True)
    args = parser.parse_args()

    n_gpus: int = torch.cuda.device_count()

    with open(args.contexts, 'r') as fp:
        contexts = fp.readlines()
    context_lists = np.array_split(contexts, n_gpus)
    context_lists = [cl for cl in context_lists if len(cl) > 0]

    os.system(f'rm -rf topic_data/tmp/')
    os.makedirs('topic_data/tmp/')
    for i, cl in enumerate(context_lists):
        with open(f'topic_data/tmp/topics_{i}.txt', 'w') as fp:
            fp.writelines(cl)
            fp.close()

    os.makedirs(args.out_dir, exist_ok=True)
    # Run generation multi-threaded (one thread per GPU)
    threads = []
    for i in range(len(context_lists)):
        th = threading.Thread(
            target=run,
            args=(i, args.num_samples, args.out_dir)
        )
        th.start()
        threads.append(th)

    for th in threads:
        th.join()
