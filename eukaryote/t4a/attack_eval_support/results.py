import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from tabulate import tabulate

from eukaryote.t4a.attack_eval_support.metrics import (
    calculate_binary_metrics,
    calculate_multiclass_metrics,
)


def default_attack_labels(n):
    return [f"Attack {i}" for i in range(1, n + 1)]


class Results:
    """Shared class to represent attack (or training) results for the purpose of
    calculating metrics, plots, saving to a file, etc.

    Motivating example: during training, it may be useful to trace training
    precision and recall vs. evaluation precision and recall, as opposed to only
    accuracy, as is done in TextAttack. Constructing `Results` instances from
    training results in comparison with instances from evaluation/attack results
    allows for easily plotting training vs. evaluation metrics across epochs.

    Args:
        attack_labels (list[str]):
            The labels of each of the attack results. Length
            `num_attack_results`.
        class_labels (list[str]):
            The labels of each of the classes. Length `num_classes`.
        y_true (np.ndarray):
            The true labels of each attack example. Shape
            `(num_attack_examples)`.
        y_original_pred (np.ndarray):
            The predicted labels of the original examples. Shape
            `(num_attack_examples)`.
        y_original_scores (np.ndarray):
            The predicted scores of the original examples. Shape
            `(num_attack_examples, num_classes)`.
        y_attack_pred (np.ndarray):
            The predicted labels of the attacked examples. Shape
            `(num_attack_results, num_attack_examples)`.
        y_attack_scores (np.ndarray):
            The predicted scores of the attacked examples. Shape
            `(num_attack_results, num_attack_examples, num_classes)`.
    """

    def __init__(
        self,
        attack_labels,
        class_labels,
        y_true,
        y_original_pred,
        y_original_scores,
        y_attack_pred,
        y_attack_scores,
        original_label="Original",
    ):
        self.attack_labels = attack_labels
        self.class_labels = class_labels
        self.y_true = y_true
        self.y_original_pred = y_original_pred
        self.y_original_scores = y_original_scores
        self.y_attack_pred = y_attack_pred
        self.y_attack_scores = y_attack_scores
        self.original_label = original_label

    def calculate_metrics(self):
        """Calculate and collect all metrics.

        Returns:
            tuple[dict, list[dict]]: The metric labels and list of metrics,
            where both the metric labels and each metrics object is a
            dictionary.
        """

        num_attack_results = self.y_attack_pred.shape[0]
        num_classes = self.y_original_scores.shape[1]

        calculate_metrics_fn = (
            calculate_binary_metrics
            if num_classes <= 2
            else calculate_multiclass_metrics
        )

        metrics_list = []

        metric_labels, metrics = calculate_metrics_fn(
            self.y_true,
            self.y_original_pred,
            self.y_original_scores,
            self.class_labels,
        )
        metrics_list.append(metrics)

        for i in range(num_attack_results):
            _, metrics = calculate_metrics_fn(
                self.y_true,
                self.y_attack_pred[i],
                self.y_attack_scores[i],
                self.class_labels,
            )
            metrics_list.append(metrics)

        return metric_labels, metrics_list

    def create_table_results(self, tablefmt="plain", floatfmt=".4f"):
        """Create a table of the result metrics using the `tabulate` package.

        Args:
            tablefmt (str):
                Passed to `tabulate.tabulate`: the format of the table. Common
                values include "plain" and "html".
            floatfmt (str):
                Passed to `tabulate.tabulate`: the format string of floats.

        Returns:
            str
        """

        metric_labels, metrics_list = self.calculate_metrics()
        table = [
            ([metric_labels[key]] + [item[key] for item in metrics_list])
            for key in metric_labels.keys()
        ]

        return tabulate(
            table,
            headers=[self.original_label] + self.attack_labels,
            tablefmt=tablefmt,
            floatfmt=floatfmt,
        )

    def create_roc_plot(self):
        """Create a plot of the ROC curves."""

        fpr, tpr, _ = sklearn.metrics.roc_curve(
            self.y_true, self.y_original_scores[:, 1]
        )
        plt.plot(fpr, tpr, label=self.original_label)

        num_attack_results = self.y_attack_scores.shape[0]
        for i in range(num_attack_results):
            fpr, tpr, _ = sklearn.metrics.roc_curve(
                self.y_true, self.y_attack_scores[i, :, 1]
            )
            plt.plot(fpr, tpr, label=self.attack_labels[i])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.show()

    def create_pr_plot(self):
        """Create a plot of the precision-recall curves."""

        fpr, tpr, _ = sklearn.metrics.precision_recall_curve(
            self.y_true, self.y_original_scores[:, 1]
        )
        plt.plot(fpr, tpr, label=self.original_label)

        num_attack_results = self.y_attack_scores.shape[0]
        for i in range(num_attack_results):
            fpr, tpr, _ = sklearn.metrics.precision_recall_curve(
                self.y_true, self.y_attack_scores[i, :, 1]
            )
            plt.plot(fpr, tpr, label=self.attack_labels[i])

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall")
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.show()

    def save(self, path):
        """Save to a file.

        Args:
            path (str):
                Path to a file.
        """

        np.savez(
            path,
            attack_labels=self.attack_labels,
            y_true=self.y_true,
            y_original_pred=self.y_original_pred,
            y_original_scores=self.y_original_scores,
            y_attack_pred=self.y_attack_pred,
            y_attack_scores=self.y_attack_scores,
            **(
                {"class_labels": self.class_labels}
                if self.class_labels is not None
                else {}
            ),
        )

    @classmethod
    def from_saved(cls, path):
        """Load a `Results` instance from a file.

        Args:
            path (str):
                Path to a file.

        Returns:
            tats4aardvarks.attack_eval.Results
        """

        with np.load(path) as data:
            return cls(
                data["attack_labels"].tolist(),
                (data["class_labels"].tolist() if "class_labels" in data else None),
                data["y_true"],
                data["y_original_pred"],
                data["y_original_scores"],
                data["y_attack_pred"],
                data["y_attack_scores"],
            )

    @classmethod
    def from_singleton(
        cls,
        y_true,
        y_preds,
        y_scores,
        class_labels=None,
        original_label="Original",
    ):
        """Create a `Results` instance from a list of predictions and scores.
        Typically used from training results.
        """
        num_attack_examples = len(y_true)
        num_classes = y_scores.shape[1]

        return cls(
            [],
            class_labels,
            y_true,
            y_preds,
            y_scores,
            np.empty((0, num_attack_examples)),
            np.empty((0, num_attack_examples, num_classes)),
            original_label,
        )

    @classmethod
    def from_attack_results(
        cls, attack_results_list, attack_labels=None, class_labels=None
    ):
        """Create a `Results` instance a list of
        `textattack.attack_results.attack_result.AttackResult`s.

        Args:
            attack_results_list (list[
                textattack.attack_results.attack_result.AttackResult]):
                List of attack results.
            attack_labels (Optional[list[str]]):
                Labels of each attack results. If not provided, defaults labels
                are generated.
            class_labels (Optional[list[str]]):
                Labels of each class. If not provided, defaults labels are
                inferred.

        Returns:
            tats4aardvarks.attack_eval.Results
        """

        example_result = attack_results_list[0][0].original_result
        num_attack_results = len(attack_results_list)
        num_attack_examples = len(attack_results_list[0])
        num_classes = example_result.raw_output.shape[0]

        if attack_labels is None:
            attack_labels = default_attack_labels(num_attack_results)
        if len(attack_labels) != num_attack_results:
            raise ValueError("Unequal number of attack labels and results")

        if class_labels is None:
            class_labels = example_result.attacked_text.attack_attrs.get("label_names")
        if class_labels is not None and len(class_labels) != num_classes:
            class_labels = None

        y_true = np.empty(num_attack_examples)
        y_original_pred = np.empty(num_attack_examples)
        y_original_scores = np.empty((num_attack_examples, num_classes))
        y_attack_pred = np.empty((num_attack_results, num_attack_examples))
        y_attack_scores = np.empty(
            (num_attack_results, num_attack_examples, num_classes)
        )

        for i, attack_results in enumerate(attack_results_list):
            for j, attack_result in enumerate(attack_results):
                if i == 0:
                    y_true[j] = attack_result.original_result.ground_truth_output
                    y_original_pred[j] = attack_result.original_result.output
                    y_original_scores[j] = attack_result.original_result.raw_output
                y_attack_pred[i, j] = attack_result.perturbed_result.output
                y_attack_scores[i, j] = attack_result.perturbed_result.raw_output

        return cls(
            attack_labels,
            class_labels,
            y_true,
            y_original_pred,
            y_original_scores,
            y_attack_pred,
            y_attack_scores,
        )
