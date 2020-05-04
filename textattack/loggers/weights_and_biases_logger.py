import wandb

from .logger import Logger

class WeightsAndBiasesLogger(Logger):
    def __init__(self, filename='', stdout=False):
        wandb.init()

    def log_attack_result(self, result):
        original_text_colored, perturbed_text_colored = result.diff_color(color_method='html')
        original_text_colored = wandb.Html(original_text_colored)
        perturbed_text_colored = wandb.Html(perturbed_text_colored)
        wandb.log({
            'type': type(result).__name__,
            'original_text': original_text_colored,
            'perturbed_text': perturbed_text_colored,
            'original_output': result.original_result.output,
            'perturbed_output': result.perturbed_result.output,
        })

    def log_summary_rows(self, rows, title, window_id):
        print('w&b skipping summary')
        pass
        # @TODO: should we log some summary to W&B? It seems to automatically
        # calculate its own summary statistics.

    def log_sep(self):
        self.fout.write('-' * 90 + '\n')
        
