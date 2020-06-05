def preprocess_ids(lists_of_ids):
    """ Tries to automatically aggregating lists of IDs produced by tokenizers. """
    lists_of_ids = unroll_tuples(lists_of_ids)
    return pad_and_truncate_lists(lists_of_ids)

def unroll_tuples(list_of_tuples):
    """ Determines if a list is a list of tuples of length 1. If so, removes
        each item from its tuple.
    """
    if not isinstance(list_of_tuples[0], tuple):
        return list_of_tuples
    elif not len(list_of_tuples[0]) == 1:
        return list_of_tuples
    return [l[0] for l in list_of_tuples]

def pad_and_truncate_lists(lists, pad_token=0):
    """ Pads lists with trailing zeros to make them all the same length. """
    max_list_len = max(len(l) for l in lists)
    for i in range(len(lists)):
        lists[i] += ([pad_token] * (max_list_len - len(lists[i])))
    return lists