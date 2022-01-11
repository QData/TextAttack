from itertools import islice
import multiprocessing
from multiprocessing import Process, Queue, cpu_count, Pool

import torch

from textattack.augmentation import (
    CharSwapAugmenter,
    CheckListAugmenter,
    CLAREAugmenter,
)


def augment_parallel_direct(text_list, augmenter, num_processes):

    # decide how many processes to start
    num_processes = min(cpu_count(), len(text_list), num_processes)
    print(f"Starting computations on {num_processes} cores....\n")

    # setting up data
    queue = Queue()
    text_list = [(idx, text) for idx, text in enumerate(text_list)]

    # set up sub lists to feed to each processes
    remainder = len(text_list) % num_processes
    result = len(text_list) // num_processes
    num_ls = [result for _ in range(num_processes)]
    for i in range(remainder):
        num_ls[i] += 1
    iter_text_list = iter(text_list)
    list_of_tx_ls = [list(islice(iter_text_list, elem)) for elem in num_ls]

    for x in list_of_tx_ls:
        print(x)

    # create processes
    processes = [Process(target=augment_from_queue, args=(tx_ls, queue, augmenter)) for tx_ls in list_of_tx_ls]

    # run augmentations
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # collect and sort the results
    unsorted_result = [queue.get() for _ in range(len(text_list))]
    result = [val[1] for val in sorted(unsorted_result)]
    return result


def augment_parallel_map(text_list, augmenter, num_processes):

    # decide how many processes to start
    num_processes = min(cpu_count(), len(text_list), num_processes)
    print(f"Starting computations on {num_processes} cores....\n")

    with Pool(num_processes) as pool:
        result = pool.map(augmenter.augment, textlist)

    return result


def augment_from_queue(tx_ls, queue, augment):

    from multiprocessing import current_process

    name = current_process().name
    print(name, "Starting...")
    for text in tx_ls:
        result = (text[0], augment.augment(text[1])[0])
        queue.put(result)
    print(name, "Exiting...")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    textlist = [f"Hello my name is Hanyu liu the {i}th" for i in range(11)]
    augmento = CheckListAugmenter()

    print(augment_parallel_map(textlist, augmento, 4))
