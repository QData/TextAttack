from textattack.shared.utils import html_table_from_rows

from .logger import Logger


class WeightsAndBiasesLogger(Logger):
    """Logs attack results to Weights & Biases."""

    def __init__(self, filename="", stdout=False):
        global wandb
        import wandb

        wandb.init(project="textattack", resume=True)
        self._result_table_rows = []

    def __setstate__(self, state):
        global wandb
        import wandb

        self.__dict__ = state
        wandb.init(project="textattack", resume=True)

    def log_summary_rows(self, rows, title, window_id):
        table = wandb.Table(columns=["Attack Results", ""])
        for row in rows:
            table.add_data(*row)
            metric_name, metric_score = row
            wandb.run.summary[metric_name] = metric_score
        wandb.log({"attack_params": table})

    def _log_result_table(self):
        """Weights & Biases doesn't have a feature to automatically aggregate
        results across timesteps and display the full table.

        Therefore, we have to do it manually.
        """
        result_table = html_table_from_rows(
            self._result_table_rows, header=["", "Original Input", "Perturbed Input"]
        )
        wandb.log({"results": wandb.Html(result_table)})

    def log_attack_result(self, result):
        original_text_colored, perturbed_text_colored = result.diff_color(
            color_method="html"
        )
        result_num = len(self._result_table_rows)
        self._result_table_rows.append(
            [
                f"<b>Result {result_num}</b>",
                original_text_colored,
                perturbed_text_colored,
            ]
        )
        result_diff_table = html_table_from_rows(
            [[original_text_colored, perturbed_text_colored]]
        )
        result_diff_table = wandb.Html(result_diff_table)
        wandb.log(
            {
                "result": result_diff_table,
                "original_output": result.original_result.output,
                "perturbed_output": result.perturbed_result.output,
            }
        )
        self._log_result_table()

    def log_sep(self):
        self.fout.write("-" * 90 + "\n")
