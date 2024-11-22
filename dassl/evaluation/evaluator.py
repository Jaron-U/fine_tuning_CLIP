import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self.label_length = len(lab2cname)
        self._correct = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        # self.thresholds = cfg.THRESHOLDS
        self.thresholds = [
            [26.5, 27.1, 27.2, 27.3, 27.4],
            [23.3, 23.4, 23.6, 23.7, 23.8],
            [27.8, 27.9, 28.1, 28.2, 28.3],
            [26.8, 26.9, 27.1, 27.2, 27.3],
            [28.3, 28.4, 28.6, 28.7, 28.8]
        ]
        self.results = {
            label: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for label in range(self.label_length)
        }
        self._results_by_threshold = {
            label: [
                {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for _ in range(len(self.thresholds[label]))
            ] for label in range(self.label_length)
        }
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self.results = {
            label: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for label in range(self.label_length)
        }
        self._results_by_threshold = {
            label: [
                {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for _ in range(len(self.thresholds[label]))
            ] for label in range(self.label_length)
        }
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        mo = mo.cpu().numpy()
        gt = gt.cpu().numpy()

        for i in range(len(mo)):
            true_label = gt[i]
            for label in range(self.label_length):
                for t_idx, threshold in enumerate(self.thresholds[label]):
                    satisfied = mo[i, label] > threshold

                    if true_label == label:
                        if satisfied:
                            self._results_by_threshold[label][t_idx]["TP"] += 1
                        else:
                            self._results_by_threshold[label][t_idx]["FN"] += 1
                    else:  # Incorrect class
                        if satisfied:
                            self._results_by_threshold[label][t_idx]["FP"] += 1
                        else:
                            self._results_by_threshold[label][t_idx]["TN"] += 1 

    def evaluate(self):
        results = OrderedDict()
        all_thresholds_results = []
        best_thresholds = [{"F1": 0, "Threshold": 0, "Precision": 0, "Recall": 0} for _ in range(self.label_length)]

        for label in range(self.label_length): 
            for t_idx in range(len(self.thresholds[label])): 
                TP = self._results_by_threshold[label][t_idx]["TP"]
                FP = self._results_by_threshold[label][t_idx]["FP"]
                FN = self._results_by_threshold[label][t_idx]["FN"]
                TN = self._results_by_threshold[label][t_idx]["TN"]

                precision = round(TP / (TP + FP), 3) if (TP + FP) > 0 else 0.0
                recall = round(TP / (TP + FN), 3) if (TP + FN) > 0 else 0.0
                f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0

                all_thresholds_results.append({
                    "Class": self._lab2cname[label],
                    "Threshold": self.thresholds[label][t_idx],
                    "TP": TP,
                    "TN": TN,
                    "FP": FP,
                    "FN": FN,
                    "Precision": precision,
                    "Recall": recall,
                    "F1_Score": f1
                })

                if f1 > best_thresholds[label]["F1"]:
                    best_thresholds[label] = {
                        "F1": f1,
                        "Threshold": self.thresholds[label][t_idx],
                        "Precision": precision,
                        "Recall": recall
                    }

        best_thresholds_results = [
            {
                "Label": self._lab2cname[label],
                "Threshold": best_thresholds[label]["Threshold"],
                "Precision": best_thresholds[label]["Precision"],
                "Recall": best_thresholds[label]["Recall"],
                "F1": best_thresholds[label]["F1"]
            }
            for label in range(self.label_length)
        ]   

        all_results_df = pd.DataFrame(all_thresholds_results)
        all_results_csv_path = osp.join(self.cfg.OUTPUT_DIR, "all_evaluate_ft_clip_metrics.csv")
        all_results_df.to_csv(all_results_csv_path, index=False)

        best_thresholds_df = pd.DataFrame(best_thresholds_results)
        best_thresholds_csv_path = osp.join(self.cfg.OUTPUT_DIR, "best_evaluate_ft_clip_metrics.csv")

        average_precision = best_thresholds_df["Precision"].mean()
        average_f1 = best_thresholds_df["F1"].mean()
        average_recall = best_thresholds_df["Recall"].mean()
        
        best_thresholds_df["average_precision"] = average_precision
        best_thresholds_df["average_recall"] = average_recall
        best_thresholds_df["average_f1"] = average_f1
        best_thresholds_df.to_csv(best_thresholds_csv_path, index=False)

        results["average_precision"] = average_precision
        results["average_recall"] = average_recall
        results["average_f1"] = average_f1

        print(
            f"Average | Precision: {average_precision:.3f} | Recall: {average_recall:.3f} | F1: {average_f1:.3f}"
        )

        return results
