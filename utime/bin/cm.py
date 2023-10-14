"""
Script to compute confusion matrices from one or more pairs of true/pred .npz
files of labels
"""

import logging
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from glob import glob
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score
from utime import Defaults
from utime.evaluation import concatenate_true_pred_pairs
from utime.evaluation import (f1_scores_from_cm, precision_scores_from_cm,
                              recall_scores_from_cm)
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Output a confusion matrix computed '
                                        'over one or more true/pred .npz '
                                        'files.')
    parser.add_argument("--true_pattern", type=str,
                        default="split*/predictions/test_data/dataset_1/files/*/true.npz",
                        help='Glob-like pattern to one or more .npz files '
                             'storing the true labels')
    parser.add_argument("--pred_pattern", type=str,
                        default="split*/predictions/test_data/dataset_1/files/*/pred.npz",
                        help='Glob-like pattern to one or more .npz files '
                             'storing the true labels')
    parser.add_argument("--normalized", action="store_true",
                        help="Normalize the CM to show fraction of total trues")
    parser.add_argument("--show_pairs", action="store_true",
                        help="Show the paired files (for debugging)")
    parser.add_argument("--group_non_rem", action="store_true",
                        help="Group all non-rem stages (N1, N2, N3) into one.")
    parser.add_argument("--round", type=int, default=3,
                        help="Round float numbers, only applicable "
                             "with --normalized.")
    parser.add_argument("--wake_trim_min", type=int, required=False,
                        help="Only evaluate on within wake_trim_min of wake "
                             "before and after sleep, as determined by true "
                             "labels")
    parser.add_argument("--period_length_sec", type=int, default=30,
                        help="Used with --wake_trim_min to determine number of"
                             " periods to trim")
    parser.add_argument("--ignore_classes", type=int, nargs="+", default=None,
                        help="Optional space separated list of class integers to ignore.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing log files.')
    parser.add_argument("--log_file", type=str, default=None,
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is None (no log file)")
    return parser


def wake_trim(pairs, wake_trim_min, period_length_sec):
    """
    Trim the pred/true pairs to remove long stretches of 'wake' in either end.
    Trims to a maximum of 'wake_trim_min' of uninterrupted 'wake' in either
    end, determined by the >TRUE< labels.

    args:
        pairs:            (list) A list of (true, prediction) pairs to trim
        wake_trim_min:    (int)  Maximum number of minutes of uninterrupted wake
                                 sleep stage (integer value '0') to allow
                                 according to TRUE values.
        period_length_sec (int)  The length in seconds of 1 period/epoch/segment

    Returns:
        List of trimmed (true, prediction) pairs
    """
    trim = int((60/period_length_sec) * wake_trim_min)
    trimmed_pairs = []
    for true, pred in pairs:
        inds = np.where(true != 0)[0]
        start = max(0, inds[0]-trim)
        end = inds[-1]+trim
        trimmed_pairs.append([
            true[start:end], pred[start:end]
        ])
    return trimmed_pairs


def trim(p1, p2):
    """
    Trims a pair of label arrays (true/pred normally) to equal length by
    removing elements from the tail of the longest array.
    This assumes that the arrays are aligned to the first element.
    """
    diff = len(p1) - len(p2)
    if diff > 0:
        p1 = p1[:len(p2)]
    else:
        p2 = p2[:len(p1)]
    return p1, p2


def glob_to_metrics_df(true_pattern: str,
                       pred_pattern: str,
                       wake_trim_min: int = None,
                       ignore_classes: list = None,
                       group_non_rem: bool = False,
                       normalized: bool = False,
                       round: int = 3,
                       period_length_sec: int = 30,
                       show_pairs: bool = False):
    """
    Run the script according to 'args' - Please refer to the argparser.
    """
    logger.info("Looking for files...")
    true = sorted(glob(true_pattern))
    pred = sorted(glob(pred_pattern))
    if not true:
        raise OSError("Did not find any 'true' files matching "
                      "pattern {}".format(true_pattern))
    if not pred:
        raise OSError("Did not find any 'true' files matching "
                      "pattern {}".format(pred_pattern))
    if len(true) != len(pred):
        raise OSError("Did not find a matching number "
                      "of true and pred files ({} and {})"
                      "".format(len(true), len(pred)))
    if len(true) != len(set(true)):
        raise ValueError("Two or more identical file names in the set "
                         "of 'true' files. Cannot uniquely match true/pred "
                         "files")
    if len(pred) != len(set(pred)):
        raise ValueError("Two or more identical file names in the set "
                         "of 'pred' files. Cannot uniquely match true/pred "
                         "files")

    pairs = list(zip(true, pred))
    if show_pairs:
        logger.info("PAIRS:\n{}".format(pairs))
    # Load the pairs
    logger.info("Loading {} pairs...".format(len(pairs)))
    l = lambda x: [np.load(f)["arr_0"] if os.path.splitext(f)[-1] == ".npz" else np.load(f) for f in x]
    np_pairs = list(map(l, pairs))
    for i, (p1, p2) in enumerate(np_pairs):
        if len(p1) != len(p2):
            logger.warning(f"Not equal lengths: {pairs[i]} {f'{len(p1)}/{len(p2)}'}. Trimming...")
            np_pairs[i] = trim(p1, p2)
    if wake_trim_min:
        logger.info("OBS: Wake trimming of {} minutes (period length {} sec)"
                    "".format(wake_trim_min, period_length_sec))
        np_pairs = wake_trim(np_pairs, wake_trim_min, period_length_sec)
    mapping = Defaults.get_class_int_to_stage_string()
    true, pred = map(lambda x: x.astype(np.uint8).reshape(-1, 1), concatenate_true_pred_pairs(pairs=np_pairs))

    labels = None
    if ignore_classes:
        labels = list((set(np.unique(true)) | set(np.unique(pred))) - set(ignore_classes))
        logger.info(f"OBS: Ignoring class(es): {ignore_classes} / {[mapping[i] for i in ignore_classes]}. "
                    f"I.e., only epochs with true labels in {labels} / {[mapping[i] for i in labels]} will be considered.")

    if group_non_rem:
        ones = np.ones_like(true)
        true = np.where(np.isin(true, [1, 2, 3]), ones, true)
        pred = np.where(np.isin(pred, [1, 2, 3]), ones, pred)
        labels.pop(labels.index(2))
        labels.pop(labels.index(3))
        mapping[1] = "NREM"
        del mapping[2]
        del mapping[3]
        logger.info(f"Merging all NREM stages into one. New labels: {labels} / {[mapping[i] for i in labels]}")

    # Print macro metrics
    keep_mask = np.where(np.isin(true, labels))
    logger.info(f"Unweighted global scores:\n"
        f"Accuracy: {np.round((true[keep_mask] == pred[keep_mask]).mean(), round)}\n"
        f"Macro F1: {np.round(f1_score(true[keep_mask], pred[keep_mask], average='macro'), round)}\n"
        f"Micro F1: {np.round(f1_score(true[keep_mask], pred[keep_mask], average='micro'), round)}\n"
        f"Kappa:    {np.round(cohen_kappa_score(true[keep_mask], pred[keep_mask]), round)}")

    cm = confusion_matrix(true, pred, labels=labels)
    if normalized:
        cm = cm.astype(np.float64)
        cm /= cm.sum(axis=1, keepdims=True)

    # Pretty print
    cm = pd.DataFrame(data=cm,
                      index=["True {}".format(mapping[i]) for i in labels],
                      columns=["Pred {}".format(mapping[i]) for i in labels])
    p = "Raw" if not normalized else "Normed"
    logger.info(f"\n\n{p} Confusion Matrix:\n" + str(cm.round(round)) + "\n")

    # Print stage-wise metrics
    f1 = f1_scores_from_cm(cm)
    prec = precision_scores_from_cm(cm)
    recall = recall_scores_from_cm(cm)
    metrics = pd.DataFrame({
        "F1": f1,
        "Precision": prec,
        "Recall/Sens.": recall
    }, index=[mapping[i] for i in labels])
    metrics = metrics.T
    metrics["mean"] = metrics.mean(axis=1)
    logger.info(f"\n\n{p} Metrics:\n" + str(np.round(metrics.T, round)) + "\n")
    return metrics


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    glob_to_metrics_df(
        true_pattern=args.true_pattern,
        pred_pattern=args.pred_pattern,
        wake_trim_min=args.wake_trim_min,
        ignore_classes=args.ignore_classes,
        group_non_rem=args.group_non_rem,
        normalized=args.normalized,
        round=args.round,
        period_length_sec=args.period_length_sec,
        show_pairs=args.show_pairs
    )


if __name__ == "__main__":
    entry_func()
