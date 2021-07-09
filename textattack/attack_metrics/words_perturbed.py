import numpy as np

from .attack_metric import AttackMetric


class WordsPerturbed(AttackMetric):
    def __init__(self, results):
        self.results = results
        self.total_attacks = len(self.results)
        self.all_num_words = np.zeros(len(self.results))
        self.perturbed_word_percentages = np.zeros(len(self.results))
        self.num_words_changed_until_success = np.zeros(2 ** 16)

        self.calculate()

    def calculate(self):
        self.max_words_changed = 0
        for i, result in enumerate(self.results):
            self.all_num_words[i] = len(result.original_result.attacked_text.words)
            num_words_changed = len(
                result.original_result.attacked_text.all_words_diff(
                    result.perturbed_result.attacked_text
                )
            )
            self.num_words_changed_until_success[num_words_changed - 1] += 1
            self.max_words_changed = max(
                self.max_words_changed or num_words_changed, num_words_changed
            )
            if len(result.original_result.attacked_text.words) > 0:
                perturbed_word_percentage = (
                    num_words_changed
                    * 100.0
                    / len(result.original_result.attacked_text.words)
                )
            else:
                perturbed_word_percentage = 0

            self.perturbed_word_percentages[i] = perturbed_word_percentage

    def avg_number_word_perturbed_num(self):
        average_num_words = self.all_num_words.mean()
        average_num_words = round(average_num_words, 2)
        return average_num_words

    def avg_perturbation_perc(self):
        self.perturbed_word_percentages = self.perturbed_word_percentages[
            self.perturbed_word_percentages > 0
        ]
        average_perc_words_perturbed = self.perturbed_word_percentages.mean()
        average_perc_words_perturbed = round(average_perc_words_perturbed, 2)
        return average_perc_words_perturbed

    def max_words_changed_num(self):
        return self.max_words_changed

    def num_words_changed_until_success_num(self):
        return self.num_words_changed_until_success
