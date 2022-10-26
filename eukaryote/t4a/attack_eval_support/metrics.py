import numpy as np
import sklearn.metrics
import sklearn.preprocessing


def calculate_binary_metrics(y_true, y_pred, y_scores, class_labels=None):
    """Calculate the binary metrics of a given result.

    Args:
        y_true (np.ndarray):
            The true labels. Shape `(num_attack_examples)`.
        y_pred (np.ndarray):
            The predicted labels. Shape
            `(num_attack_examples)`.
        y_scores (np.ndarray):
            The predicted scores. Shape
            `(num_attack_examples, num_classes)`.
        class_labels (Optional[list[str]]):
            Ignored.

    Returns:
        tuple[dict, dict]: The metric labels and values as dictionaries, where
        the keys are symbolic names of each metric.
    """

    metric_labels, metrics = {}, {}

    metric_labels["accuracy"] = "Accuracy"
    metrics["accuracy"] = sklearn.metrics.accuracy_score(y_true, y_pred)

    metric_labels["precision"] = "Precision"
    metrics["precision"] = sklearn.metrics.precision_score(y_true, y_pred)

    metric_labels["recall"] = "Recall"
    metrics["recall"] = sklearn.metrics.recall_score(y_true, y_pred)

    metric_labels["f1_score"] = "F1 Score"
    metrics["f1_score"] = sklearn.metrics.f1_score(y_true, y_pred)

    metric_labels["roc_auc"] = "ROC AUC"
    metrics["roc_auc"] = sklearn.metrics.roc_auc_score(y_true, y_scores[:, 1])

    metric_labels["average_precision"] = "Average Precision"
    metrics["average_precision"] = sklearn.metrics.average_precision_score(
        y_true, y_pred
    )

    return metric_labels, metrics


def calculate_multiclass_metrics(y_true, y_pred, y_scores, class_labels=None):
    """Calculate the multiclass metrics of a given result.

    Args:
        y_true (np.ndarray):
            The true labels. Shape `(num_attack_examples)`.
        y_pred (np.ndarray):
            The predicted labels. Shape
            `(num_attack_examples)`.
        y_scores (np.ndarray):
            The predicted scores. Shape
            `(num_attack_examples, num_classes)`.
        class_labels (Optional[list[str]]):
            Labels for each class by index.

    Returns:
        tuple[dict, dict]: The metric labels and values as dictionaries, where
        the keys are symbolic names of each metric.
    """

    def class_label(i):
        if class_labels is not None:
            return class_labels[i]
        return i

    metric_labels, metrics = {}, {}

    # Helper ndarrays
    num_classes = y_scores.shape[1]
    Y_true = sklearn.preprocessing.label_binarize(
        y_true, classes=np.arange(num_classes)
    )
    Y_pred = sklearn.preprocessing.label_binarize(
        y_pred, classes=np.arange(num_classes)
    )
    y_true_count = np.count_nonzero(Y_true, axis=0)
    y_pred_count = np.count_nonzero(Y_pred, axis=0)
    y_true_present = y_true_count != 0
    y_pred_present = y_pred_count != 0

    # Accuracy
    metric_labels["accuracy"] = "Accuracy"
    metrics["accuracy"] = sklearn.metrics.accuracy_score(y_true, y_pred)

    # Precision
    scores = sklearn.metrics.precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    scores.resize(num_classes)

    metric_labels["precision_weighted"] = "Precision [weighted avg]"
    metrics["precision_weighted"] = np.average(scores, weights=y_true_count)

    scores[~y_true_present] = np.nan
    for i in range(num_classes):
        metric_labels[f"precision_class{i}"] = f"Precision (class={class_label(i)})"
        metrics[f"precision_class{i}"] = scores[i]

    # Recall
    scores = sklearn.metrics.recall_score(y_true, y_pred, average=None, zero_division=0)
    scores.resize(num_classes)

    metric_labels["recall_weighted"] = "Recall [weighted avg]"
    metrics["recall_weighted"] = np.average(scores, weights=y_true_count)

    scores[~y_pred_present] = np.nan
    for i in range(num_classes):
        metric_labels[f"recall_class{i}"] = f"Recall (class={class_label(i)})"
        metrics[f"recall_class{i}"] = scores[i]

    # F1 score
    scores = sklearn.metrics.recall_score(y_true, y_pred, average=None, zero_division=0)
    scores.resize(num_classes)

    metric_labels["f1_score_weighted"] = "F1 Score [weighted avg]"
    metrics["f1_score_weighted"] = np.average(scores, weights=y_true_count)

    scores[~y_true_present & ~y_pred_present] = np.nan
    for i in range(num_classes):
        metric_labels[f"f1_score_class{i}"] = f"F1 Score (class={class_label(i)})"
        metrics[f"f1_score_class{i}"] = scores[i]

    # ROC AUC
    metric_labels["roc_auc_weighted_ovr"] = "ROC AUC [weighted avg] [one-vs-rest]"
    metric_labels["roc_auc_weighted_ovo"] = "ROC AUC [weighted avg] [one-vs-one]"
    if np.all(y_true_present):
        metrics["roc_auc_weighted_ovr"] = sklearn.metrics.roc_auc_score(
            y_true, y_scores, average="weighted", multi_class="ovr"
        )
        metrics["roc_auc_weighted_ovo"] = sklearn.metrics.roc_auc_score(
            y_true, y_scores, average="weighted", multi_class="ovo"
        )
    else:
        metrics["roc_auc_weighted_ovr"] = np.nan
        metrics["roc_auc_weighted_ovo"] = np.nan
    for i in range(num_classes):
        metric_labels[f"roc_auc_class{i}"] = f"ROC AUC (class={class_label(i)})"
        if y_true_present[i]:
            fpr, tpr, _ = sklearn.metrics.roc_curve(Y_true[:, i], y_scores[:, i])
            metrics[f"roc_auc_class{i}"] = sklearn.metrics.auc(fpr, tpr)
        else:
            metrics[f"roc_auc_class{i}"] = np.nan

    # Average precision
    scores = np.zeros(num_classes)
    for i in range(num_classes):
        if y_true_present[i]:
            scores[i] = sklearn.metrics.average_precision_score(
                Y_true[:, i], y_scores[:, i]
            )

    metric_labels["average_precision_weighted"] = "Average Precision [weighted avg]"
    metrics["average_precision_weighted"] = np.average(scores, weights=y_true_count)

    scores[~y_true_present] = np.nan
    for i in range(num_classes):
        metric_labels[
            f"average_precision_class{i}"
        ] = f"Average Precision (class={class_label(i)})"
        metrics[f"average_precision_class{i}"] = scores[i]

    return metric_labels, metrics
