import textattack
import textattack.datasets as datasets
import random
import pandas as pd
from textattack.transformations.word_swap_embedding import WordSwapEmbedding as WordSwapEmbedding
from textattack.constraints.semantics.word_embedding_distance import WordEmbeddingDistance as WordEmbeddingDistance

NUM_NEAREST = 5
NUM_SYMS = 1000

def main():
    datasets_names = [datasets.classification.AGNews, datasets.classification.IMDBSentiment, datasets.classification.MovieReviewSentiment, datasets.classification.YelpSentiment]
    cand_words = list()
    swap = WordSwapEmbedding(max_candidates=NUM_NEAREST)
    for name in datasets_names:
        data = name()
        for label, text in data:
            words = textattack.tokenized_text.raw_words(text)
            for word in words:
                if word.lower() in swap.stopwords:
                    continue
                cand_words.append(word)
    random.shuffle(cand_words)
    df = pd.DataFrame()
    dist_calc = WordEmbeddingDistance()
    num_added = 0
    added_words = set()
    for i, word in enumerate(cand_words):
        if num_added == NUM_SYMS:
            break
        word = word.lower()
        if word in added_words:
            continue
        replacement_words = swap._get_replacement_words(word)
        if len(replacement_words) < NUM_NEAREST:
            continue
        replacement_word = replacement_words[i % NUM_NEAREST].lower()
        try:
            w_id = dist_calc.word_embedding_word2index[word]
            rw_id = dist_calc.word_embedding_word2index[replacement_word]
        except KeyError:
            continue
        cos_sim = float(dist_calc.get_cos_sim(w_id, rw_id))
        mse_dist = float(dist_calc.get_mse_dist(w_id, rw_id))
        row = {'word_a': word, 'word_b': replacement_word, 'cos_sim': cos_sim, 'mse_dist': mse_dist}
        df = df.append(row, ignore_index=True)
        added_words.add(word)
        num_added += 1
    df.to_csv('examples.csv')

if __name__ == '__main__':
    main()
