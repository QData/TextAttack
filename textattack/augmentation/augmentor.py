from itertools import islice
import multiprocessing
from multiprocessing import Process, Queue, cpu_count, Pool
import torch
from .augmenter import Augmenter
import os, sys
import os
import argparse
import datasets
import tqdm
import torch
import random
import numpy as np


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Augmentor:

    def __init__(self, augmenter, dataset, augment_args=None, seed=None):

        assert isinstance(
            augmenter, Augmenter
        ), f"`augmenter` argument must be of type `textattack.Augmenter`, but got type of `{type(augmenter)}`."
        self.augmenter = augmenter
        self.dataset = dataset
        self.augment_args = augment_args
        self.seed = seed

    def augment(self):
        return self.augmenter.augment_many(self.dataset)

    def augment_parallel_lazy(self, num_processes):
        # decide how many processes to start
        num_processes = min(cpu_count(), len(self.dataset), num_processes)
        print(f"Starting computations on {num_processes} cores....\n")

        with HiddenPrints():
            # setting up data
            queue = Queue()
            dataset = [(idx, text) for idx, text in enumerate(self.dataset)]

            # set up sub lists to feed to each processes
            remainder = len(dataset) % num_processes
            div = len(dataset) // num_processes
            num_ls = [div for _ in range(num_processes)]
            for i in range(remainder):
                num_ls[i] += 1
            iter_text_list = iter(dataset)
            list_of_tx_ls = [list(islice(iter_text_list, elem)) for elem in num_ls]

            '''
            for x in list_of_tx_ls:
                print(x)
            '''

            # create processes
            processes = [Process(target=augment_from_queue, args=(tx_ls, queue, self.augmenter)) for tx_ls in
                         list_of_tx_ls]

            # run augmentations
            for p in processes:
                p.start()
            for p in processes:
                p.join()

        # collect and sort the results
        unsorted_result = [queue.get() for _ in range(len(dataset))]
        result = [val[1] for val in sorted(unsorted_result)]
        return result

    def augment_parallel_map(self, num_processes):

        # decide how many processes to start
        num_processes = min(cpu_count(), len(self.dataset), num_processes)
        print(f"Starting computations on {num_processes} cores....\n")

        with Pool(num_processes) as pool:
            result = pool.map(self.augmenter.augment, self.dataset)

        return result

    def augment_parallel_cpu(self, num_processes):
        augmented_text = []
        augmented_indices = []
        num_workers = min(cpu_count(), num_processes)
        assert num_workers >= 1, "You need at least one GPU to perform augmentation."

        """
        torch.multiprocessing.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
        """

        in_queue = torch.multiprocessing.Queue()
        out_queue = torch.multiprocessing.Queue()
        for i, text in enumerate(self.dataset):
            in_queue.put((i, text))

        # Start workers.
        worker_pool = torch.multiprocessing.Pool(
            num_workers,
            augment_from_queue_cpu,
            (
                in_queue,
                out_queue,
                self.augmenter
            ),
        )
        pbar = tqdm.tqdm(total=len(self.dataset), smoothing=0)
        for _ in range(len(self.dataset)):
            idx, aug_text = out_queue.get(block=True)
            pbar.update()
            if isinstance(aug_text, Exception):
                continue
            if aug_text == "":
                continue
            augmented_indices.append(idx)
            augmented_text.append(aug_text)

        # Send sentinel values to worker processes
        for _ in range(num_workers):
            in_queue.put(("END", "END"))
        worker_pool.terminate()
        worker_pool.join()

        augmented_indices = np.array(augmented_indices)
        argsort_indices = np.argsort(augmented_indices)
        augmented_text = [augmented_text[i] for i in argsort_indices]

        return augmented_text

    def augment_parallel_gpu(self, num_processes):
        augmented_text = []
        augmented_indices = []
        num_workers = min(torch.cuda.device_count(), num_processes)

        """
        torch.multiprocessing.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")
        """

        in_queue = torch.multiprocessing.Queue()
        out_queue = torch.multiprocessing.Queue()
        for i, text in enumerate(self.dataset):
            in_queue.put((i, text))

        # Start workers.
        worker_pool = torch.multiprocessing.Pool(
            num_workers,
            augment_from_queue_gpu,
            (
                num_workers,
                in_queue,
                out_queue,
                self.seed,
                self.augmenter,
            ),
        )
        pbar = tqdm.tqdm(total=len(self.dataset), smoothing=0)
        for _ in range(len(self.dataset)):
            idx, aug_text = out_queue.get(block=True)
            pbar.update()
            if isinstance(aug_text, Exception):
                continue
            if aug_text == "":
                continue
            augmented_indices.append(idx)
            augmented_text.append(aug_text)

        # Send sentinel values to worker processes
        for _ in range(num_workers):
            in_queue.put(("END", "END"))
        worker_pool.terminate()
        worker_pool.join()

        augmented_indices = np.array(augmented_indices)
        argsort_indices = np.argsort(augmented_indices)
        augmented_text = [augmented_text[i] for i in argsort_indices]

        return augmented_text


def augment_from_queue(tx_ls, queue, augment):
    for text in tx_ls:
        result = (text[0], augment.augment(text[1])[0])
        queue.put(result)


def augment_from_queue_cpu(in_queue, out_queue, augmenter):

    while True:
        try:
            i, inputs = in_queue.get()
            if i == "END" and inputs == "END":
                # End process when sentinel value is received
                break
            else:
                if isinstance(inputs, tuple):
                    text_to_augment = inputs[1]
                else:
                    text_to_augment = inputs

                augmented_text = augmenter.augment(text_to_augment)

                out_queue.put((i, augmented_text))
        except Exception as e:
            out_queue.put((i, e))


def augment_from_queue_gpu(num_gpus, in_queue, out_queue, seed, augmenter):
    gpu_id = (torch.multiprocessing.current_process()._identity[0] - 1) % num_gpus
    set_seed(seed)
    torch.cuda.set_device(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)

    while True:
        try:
            i, inputs = in_queue.get()
            if i == "END" and inputs == "END" :
                # End process when sentinel value is received
                break
            else:
                if isinstance(inputs, tuple):
                    text_to_augment = inputs[1]
                else:
                    text_to_augment = inputs

                augmented_text = augmenter.augment(text_to_augment)

                out_queue.put((i, augmented_text))
        except Exception as e:
            out_queue.put((i, e))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
