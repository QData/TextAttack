"""
Takes in a list of words and their vectors and preprocesses them. Outputs a list of vectors,
    a list of words, and a list of the nearest neighbors for each word with respect
    to this embedding.
example usage: 
    python preprocess_glove.py --w word_embeddings/retrofit/ --i glove_retrofit.txt
"""

import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.neighbors import KDTree

parser = argparse.ArgumentParser()
parser.add_argument('--word_dir', '--w', default='glove/')
parser.add_argument('--input_file', '--i', default='glove.6B.100d.txt')
parser.add_argument('--output_file', '--o', default='glove_100d.npy')
parser.add_argument('--word_list_outfile', '--e', default='wordlist.pickle')
parser.add_argument('--nn_file', '--n', default='glovenn.npy')
args = parser.parse_args()

WORD_DIR = args.word_dir
input_file_name = args.input_file
output_file_name = args.output_file
word_list_outfile_name = args.word_list_outfile
nn_file_name = args.nn_file

print('Preprocessing {} to {}.'.format(input_file_name, \
        [output_file_name, word_list_outfile_name, nn_file_name]))

input_file_name = os.path.join(WORD_DIR, input_file_name)
output_file_name = os.path.join(WORD_DIR, output_file_name)
word_list_outfile_name = os.path.join(WORD_DIR, word_list_outfile_name)
nn_file_name = os.path.join(WORD_DIR, nn_file_name)

embeddings_index = {}
embeddings = []
f = open(input_file_name)
i = 0 
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = i
    embeddings.append(coefs)
    i+=1
embedding_matrix = np.array(embeddings)
print(embedding_matrix)
print(embedding_matrix.shape)

np.save(open(output_file_name,'wb'),embedding_matrix)
pickle.dump(embeddings_index,open(word_list_outfile_name,'wb'))

nbrs = KDTree(embedding_matrix)
print('tree finished')
indices = []
for i in tqdm(range(embedding_matrix.shape[0]//10)):
    dist, ind = nbrs.query(embedding_matrix[i*10:i*10+10],k = 101)
    indices.append(ind)
indices = np.concatenate(indices, axis = 0)
np.save(open(nn_file_name,'wb'),indices[:,:])