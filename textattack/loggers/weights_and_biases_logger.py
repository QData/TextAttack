import wandb

from textattack.shared.utils import html_table_from_rows
from .logger import Logger

class WeightsAndBiasesLogger(Logger):
    def __init__(self, filename='', stdout=False):
        wandb.init(project='textattack')
        self._result_table_rows = []
        
    def _log_result_table(self):
        """ Weights & Biases doesn't have a feature to automatically 
            aggregate results across timesteps and display the full table.
            Therefore, we have to do it manually.
        """
        result_table = html_table_from_rows(self._result_table_rows)
        wandb.log({ 'results': wandb.Html(result_table) })
        
    def log_attack_result(self, result):
        original_text_colored, perturbed_text_colored = result.diff_color(color_method='html')
        self._result_table_rows.append([original_text_colored, perturbed_text_colored])
        result_diff_table = html_table_from_rows([[original_text_colored, perturbed_text_colored]])
        result_diff_table = wandb.Html(result_diff_table)
        wandb.log({
            'resultType': type(result).__name__,
            'result': result_diff_table,
            'original_output': result.original_result.output,
            'perturbed_output': result.perturbed_result.output,
        })
        self._log_result_table()

    def log_summary_rows(self, rows, title, window_id):
        print('w&b skipping summary')
        pass
        # @TODO: should we log some summary to W&B? It seems to automatically
        # calculate its own summary statistics.

    def log_sep(self):
        self.fout.write('-' * 90 + '\n')
        
