from .attack_metrics import AttackMetric

class WordsPerturbed(AttackMetric):
	def __init__(results):
        self.total_attacks = len(self.results)
        self.all_num_words = np.zeros(len(self.results))
        self.perturbed_word_percentages = np.zeros(len(self.results))

	def calculate():
        num_words_changed_until_success = np.zeros(
            2 ** 16
        ) 
        max_words_changed = 0
		for i, self.result in enumerate(self.results):
			self.all_num_words[i] = len(result.original_result.attacked_text.words)
            num_words_changed = len(
                result.original_result.attacked_text.all_words_diff(
                    result.perturbed_result.attacked_text
                )
            )
            num_words_changed_until_success[num_words_changed - 1] += 1
            max_words_changed = max(
                max_words_changed or num_words_changed, num_words_changed
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


	def avg_number_word_perturbed():
		average_num_words = all_num_words.mean()
        average_num_words = round(average_num_words, 2)
        return average_num_words

	def perturbation_percentage():
		self.perturbed_word_percentages = self.perturbed_word_percentages[
            self.perturbed_word_percentages > 0
        ]
        average_perc_words_perturbed = self.perturbed_word_percentages.mean()
        # average_perc_words_perturbed = str(round(average_perc_words_perturbed, 2)) + "%"
        average_perc_words_perturbed = round(average_perc_words_perturbed, 2)
        return average_perc_words_perturbed

        
		