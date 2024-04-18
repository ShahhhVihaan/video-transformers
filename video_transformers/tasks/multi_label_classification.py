from typing import Any, List

import evaluate
import torch

from video_transformers.tasks.base import TaskMixin


class Combine:
    def __init__(self, metrics: List[str]):
        self.metrics = [evaluate.load(metric) if isinstance(metric, str) else metric for metric in metrics]

    def add_batch(self, predictions: Any, references: Any):
        for metric in self.metrics:
            metric.add_batch(predictions=predictions, references=references)

    def compute(self, **kwargs):
        results = {}
        zero_division = kwargs.get("zero_division", "warn")
        kwargs.pop("zero_division")
        for metric in self.metrics:
            if metric.name == "precision":
                results.update(metric.compute(zero_division=zero_division, **kwargs))
            else:
                results.update(metric.compute(**kwargs))
        return results


class MultiLabelClassificationTaskMixin(TaskMixin):
    def __init__(self, *args, **kwargs):
        super(MultiLabelClassificationTaskMixin, self).__init__(*args, **kwargs)
        self._train_metrics = Combine([evaluate.load("f1", "multi"), evaluate.load("precision", "multi"), evaluate.load("recall", "multi")])
        self._val_metrics = Combine([evaluate.load("f1", "multi"), evaluate.load("precision", "multi"), evaluate.load("recall", "multi")])
        self._last_train_result = None 
        self._last_val_result = None
        self.task = "multi_label_classification"

    @property
    def train_metrics(self):
        return self._train_metrics

    @property
    def val_metrics(self):
        return self._val_metrics

    @property
    def last_train_result(self):
        return self._last_train_result

    @property
    def last_val_result(self):
        return self._last_val_result

    def training_step(self, batch):
        inputs = batch["video"]
        labels = batch["label"]
        outputs = self.model(inputs)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).float()
        all_predictions = self.accelerator.gather(predictions)
        all_labels = self.accelerator.gather(labels)
        self.train_metrics.add_batch(predictions=all_predictions, references=all_labels)
        loss = self.loss_function(outputs, labels)
        return loss

    def on_training_epoch_end(self):
        if self.accelerator.is_main_process:
            result = self.train_metrics.compute()
            for metric, score in result.items():
                self.accelerator.log({f"train/{metric}": score}, step=self.overall_epoch)

            self._last_train_result = result

    def validation_step(self, batch):
        inputs = batch["video"]
        labels = batch["label"]
        outputs = self.model(inputs)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).float()
        all_predictions = self.accelerator.gather(predictions)
        all_labels = self.accelerator.gather(labels)
        self.val_metrics.add_batch(predictions=all_predictions, references=all_labels)
        loss = self.loss_function(outputs, labels)
        return loss

    def on_validation_epoch_end(self):
        if self.accelerator.is_main_process:
            result = self.val_metrics.compute()
            for metric, score in result.items():
                self.accelerator.log({f"val/{metric}": score}, step=self.overall_epoch)

            self._last_val_result = result
