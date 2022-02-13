from itertools import islice
import multiprocessing
from multiprocessing import Process, Queue, cpu_count, Pool
import torch
from .augmenter import Augmenter
import os, sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Augmentor:

    def __init__(self, augmenter, dataset, augment_args=None):

        assert isinstance(
            augmenter, Augmenter
        ), f"`augmenter` argument must be of type `textattack.Augmenter`, but got type of `{type(augmenter)}`."
        self.augmenter = augmenter
        self.dataset = dataset
        self.augment_args = augment_args

    def augment(self):
        return self.augmenter.augment_many(self.dataset)

    def augment_parallel(self, num_processes):
        # decide how many processes to start
        num_processes = min(cpu_count(), len(self.dataset), num_processes)
        print(f"Starting computations on {num_processes} cores....\n")

        with HiddenPrints():
            # setting up data
            queue = Queue()
            dataset = [(idx, text) for idx, text in enumerate(self.dataset)]

            # set up sub lists to feed to each processes
            remainder = len(dataset) % num_processes
            result = len(dataset) // num_processes
            num_ls = [result for _ in range(num_processes)]
            for i in range(remainder):
                num_ls[i] += 1
            iter_text_list = iter(dataset)
            list_of_tx_ls = [list(islice(iter_text_list, elem)) for elem in num_ls]

            for x in list_of_tx_ls:
                print(x)

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


def augment_from_queue(tx_ls, queue, augment):

    from multiprocessing import current_process

    name = current_process().name
    #print(name, "Starting...")
    for text in tx_ls:
        result = (text[0], augment.augment(text[1])[0])
        queue.put(result)
    #print(name, "Exiting...")



