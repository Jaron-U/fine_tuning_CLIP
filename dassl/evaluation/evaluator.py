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
        self.thresholds = cfg.THRESHOLDS
        self.results = {
            label: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for label in range(self.label_length)
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
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        mo = mo.cpu().numpy()
        gt = gt.cpu().numpy()

        for i in range(len(mo)):
            true_label = gt[i]
            satified_labels = [
                label 
                for label in range(self.label_length) 
                if mo[i, label] > self.thresholds[label]
            ]

            if not satified_labels:
                satified_labels = [self.label_length]
            
            self._y_true.append(true_label)
            self._y_pred.append(satified_labels)

            for label in range(self.label_length):
                if true_label == label:
                    if label in satified_labels:
                        self.results[label]["TP"] += 1
                    else:
                        self.results[label]["FN"] += 1
                else:
                    if label in satified_labels:
                        self.results[label]["FP"] += 1
                    else:
                        self.results[label]["TN"] += 1

        # pred = mo.max(1)[1]
        # matches = pred.eq(gt).float()
        # self._correct += int(matches.sum().item())
        # self._total += gt.shape[0]

        # self._y_true.extend(gt.data.cpu().numpy().tolist())
        # self._y_pred.extend(pred.data.cpu().numpy().tolist())

        # if self._per_class_res is not None:
        #     for i, label in enumerate(gt):
        #         label = label.item()
        #         matches_i = int(matches[i].item())
        #         self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        precision_list = []
        recall_list = []
        f1_list = []

        for label in range(self.label_length):
            TP = self.results[label]["TP"]
            FP = self.results[label]["FP"]
            FN = self.results[label]["FN"]
            TN = self.results[label]["TN"]

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

            f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

            class_name = self._lab2cname[label]
            print(
                f"Class: {class_name} | TP: {TP} | FP: {FP} | FN: {FN} | TN: {TN} | "
                f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}"
            )
        
        df = pd.DataFrame.from_dict(self.results, orient="index")
        df["Precision"] = precision_list
        df["Recall"] = recall_list
        df["F1"] = f1_list
        df["Average_Precision"] = df["Precision"].mean()
        df["Average_Recall"] = df["Recall"].mean()
        df["Average_F1"] = df["F1"].mean()

        csv_path = osp.join(self.cfg.OUTPUT_DIR, "evaluate_ft_clip_metrics.csv")
        df.to_csv(csv_path, index=False)

        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            self._y_true,
            [pred[0] if isinstance(pred, list) else pred for pred in self._y_pred],
            average="macro",
            labels=range(5),
        )

        results["overall_precision"] = overall_precision * 100
        results["overall_recall"] = overall_recall * 100
        results["overall_f1"] = overall_f1 * 100
        print(
            f"Overall | Precision: {overall_precision:.3f} | Recall: {overall_recall:.3f} | F1: {overall_f1:.3f}"
        )

        return results


        # results = OrderedDict()
        # acc = 100.0 * self._correct / self._total
        # err = 100.0 - acc
        # macro_f1 = 100.0 * f1_score(
        #     self._y_true,
        #     self._y_pred,
        #     average="macro",
        #     labels=np.unique(self._y_true)
        # )

        # # The first value will be returned by trainer.test()
        # results["accuracy"] = acc
        # results["error_rate"] = err
        # results["macro_f1"] = macro_f1

        # print(
        #     "=> result\n"
        #     f"* total: {self._total:,}\n"
        #     f"* correct: {self._correct:,}\n"
        #     f"* accuracy: {acc:.1f}%\n"
        #     f"* error: {err:.1f}%\n"
        #     f"* macro_f1: {macro_f1:.1f}%"
        # )

        # if self._per_class_res is not None:
        #     labels = list(self._per_class_res.keys())
        #     labels.sort()

        #     print("=> per-class result")
        #     accs = []

        #     for label in labels:
        #         classname = self._lab2cname[label]
        #         res = self._per_class_res[label]
        #         correct = sum(res)
        #         total = len(res)
        #         acc = 100.0 * correct / total
        #         accs.append(acc)
        #         print(
        #             f"* class: {label} ({classname})\t"
        #             f"total: {total:,}\t"
        #             f"correct: {correct:,}\t"
        #             f"acc: {acc:.1f}%"
        #         )
        #     mean_acc = np.mean(accs)
        #     print(f"* average: {mean_acc:.1f}%")

        #     results["perclass_accuracy"] = mean_acc

        # if self.cfg.TEST.COMPUTE_CMAT:
        #     cmat = confusion_matrix(
        #         self._y_true, self._y_pred, normalize="true"
        #     )
        #     save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
        #     torch.save(cmat, save_path)
        #     print(f"Confusion matrix is saved to {save_path}")

        # return results
