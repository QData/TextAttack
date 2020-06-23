import functools
import math

import numpy as np
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

import textattack
from textattack.search_methods import SearchMethod
from textattack.shared import utils


class MetropolisHastingsSampling(SearchMethod):
    """ 
    Uses Metropolis-Hastings Sampling to generate adversarial samples.
    Based off paper "Generating Fluent Adversarial Examples for Natural Langauges" by Zhang, Zhou, Miao, Li (2019)

    Args:
        max_iter (int): The maximum number of sampling to perform. 
            If the word count of the text under attack is greater than `max_iter`, we replace max_iter with word count for that specific example.
            This is so we at least try to perturb every word once. 
        lm_type (str): The language model to use to estimate likelihood of text.
            Currently supported LM is "gpt-2"
    """

    def __init__(self, max_iter=200, lm_type="gpt2"):
        self.max_iter = max_iter
        self.lm_type = lm_type

        self._lm_tokenizer = AutoTokenizer.from_pretrained(self.lm_type)
        self._language_model = AutoModelWithLMHead.from_pretrained(self.lm_type)

        try:
            # Try to use GPU, but sometimes we might have out-of-memory issue
            # Having the below line prevents CUBLAS error when we try to switch to CPU
            torch.cuda.current_blas_handle()
            self._lm_device = utils.get_device()
            self._language_model = self._language_model.to(self._lm_device)
        except RuntimeError as error:
            if "CUDA out of memory" in str(error):
                textattack.shared.utils.get_logger().warn(
                    "CUDA out of memory. Running GPT-2 for Metropolis Hastings on CPU."
                )
                self._lm_device = torch.device("cpu")
                self._language_model = self._language_model.to(self._lm_device)
            else:
                raise error

        self._language_model.eval()

    @functools.lru_cache(maxsize=2 ** 14)
    def _lm_score(self, text):
        """
        Assigns likelihood of a text as 1/perplexity(text)
        Args:
            text (str)
        Returns: 1/perplexity(text)
        """

        input_tokens = self._lm_tokenizer.tokenize(
            text, max_length=self._lm_tokenizer.model_max_length - 2
        )
        # Occasionally, len(input_tokens) != 1022, so we have to check it
        if len(input_tokens) != self._lm_tokenizer.model_max_length - 2:
            input_tokens = input_tokens[: self._lm_tokenizer.model_max_length - 2]
        input_tokens.insert(0, self._lm_tokenizer.bos_token)
        input_tokens.append(self._lm_tokenizer.eos_token_id)
        input_ids = self._lm_tokenizer.convert_tokens_to_ids(input_tokens)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to(self._lm_device)

        with torch.no_grad():
            loss = self._language_model(input_ids, labels=input_ids)[0].item()
            del input_ids

        perplexity = math.exp(loss)
        return 1 / perplexity

    def _stationary_dist(self, x, original_output):
        """
        Unnormalized estimation of the distribution that we want to sample from.
        Args:
            x (TokenizedText)
            original_output
        Returns: lm_score(x) * prob_model(wrong|x)
        """
        model_result = self.get_goal_results([x], original_output)[0][0]
        lm_score = self._lm_score(x.text)
        return lm_score * model_result.score

    def _batch_stationary_dist(self, x_list, original_output):
        """
        Unnormalized estimation of the distribution that we want to sample from.
        Process items in batch.
        Args:
            x_list (list): List of TokenizedText
            original_output
        Returns: list of floats representing lm_score(x) * prob_model(wrong|x)
        """
        batch_size = len(x_list)
        model_results = self.get_goal_results(x_list, original_output)[0]
        lm_scores = [self._lm_score(x.text) for x in x_list]
        scores = [lm_scores[i] * model_results[i].score for i in range(batch_size)]

        return scores, model_results

    def _normalize(self, values):
        """
        Take list of values and normalize it into a probability distribution
        """
        s = sum(values)
        if s == 0:
            return [1 / len(values) for v in values]
        else:
            return [v / s for v in values]

    def _perform_search(self, initial_result):
        text_len = len(initial_result.tokenized_text.words)
        max_iter = max(self.max_iter, text_len)

        current_result = initial_result
        current_text = initial_result.tokenized_text
        current_score = self._stationary_dist(current_text, initial_result.output)

        for n in range(max_iter):
            # i-th word we want to transform
            i = n % text_len

            transformations = self.get_transformations(
                current_text,
                indices_to_modify=[i],
                original_text=initial_result.tokenized_text,
            )

            if len(transformations) == 0:
                continue

            scores, model_results = self._batch_stationary_dist(
                transformations, initial_result.output
            )
            proposal_dist = self._normalize(scores)
            # Choose one transformation randomly according to proposal distribution
            jump = np.random.choice(list(range(len(transformations))), p=proposal_dist)

            # Now we have calculate probability of return proposal g(x'|x)
            reverse_transformations = self.get_transformations(
                transformations[jump],
                indices_to_modify=[i],
                original_text=initial_result.tokenized_text,
            )

            reverse_jump = None
            for k in range(len(reverse_transformations)):
                if reverse_transformations[k].text == current_text.text:
                    # Transition x -> x' exists
                    reverse_jump = k
                    break
            if not reverse_jump:
                return_prob = 0
            else:
                ret_scores, _ = self._batch_stationary_dist(
                    reverse_transformations, initial_result.output
                )
                return_prob = self._normalize(ret_scores)[reverse_jump]

            """
            According to Metropolis-Hastings algorithm
            let f(x) be value proportional to target distribution p(x)
            and g(x|x') be transition probability from x' to x.
            Then, acceptance ratio = min(1, (f(x')*g(x|x')) / (f(x) * g(x'|x)))
            """
            if current_score == 0.0:
                current_score += 1e-9
            acceptance_ratio = (scores[jump] * return_prob) / (
                current_score * proposal_dist[jump]
            )
            acceptance_ratio = min(1, acceptance_ratio)
            u = np.random.uniform(low=0.0, high=1.0)

            if acceptance_ratio >= u or model_results[jump].succeeded:
                # Accept the proposed jump
                current_result = model_results[jump]
                current_text = transformations[jump]
                current_score = scores[jump]

            if current_result.succeeded:
                break

        return current_result

    def extra_repr_keys(self):
        return ["max_iter", "lm_type"]
