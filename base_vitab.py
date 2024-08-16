import csv
import enum
import os
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torchmetrics
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.optim import optimizer
from torch.utils import data
from tqdm import tqdm


def plot_model_performance(
    measure_kinds: tuple[str, str],
    model_name: str,
    plotsets: list[tuple[list[float], str, str]],
    second_plotsets: list[tuple[list[float], str, str]] = None,
):
    _, ax1 = plt.subplots(figsize=(15, 8))
    plot_legend = []

    # Plot the primary data
    ax1.set_title(
        f"{model_name} Performance",
        fontsize="x-large",
        pad=15,
    )
    ax1.set_xlabel("Epochs", fontsize="large")
    ax1.set_ylabel(measure_kinds[0], fontsize="large", color=plotsets[0][2])

    for performance_by_time, line_label, color in plotsets:
        x_values = np.arange(1, len(performance_by_time) + 1)
        (plot,) = ax1.plot(
            x_values,
            performance_by_time,
            # marker="o",
            label=line_label,
            color=color,
        )
        plot_legend.append(plot)
        ax1.set_xlim(-0.3, len(performance_by_time) + 0.3)

    ax1.locator_params(axis="y", nbins=30, tight=False)
    ax1.locator_params(axis="x", nbins=len(plotsets[0][0]) + 2, tight=False)
    ax1.tick_params(axis="y")
    ax1.grid(color="gray", linestyle="--", linewidth=0.5)

    # Plot secondary datasets with unique y-axis scale.
    if second_plotsets:
        ax2 = ax1.twinx()
        ax2.set_ylabel(measure_kinds[1], fontsize="large", color=second_plotsets[0][2])
        for performance_by_time, line_label, color in second_plotsets:
            x_values = np.arange(1, len(performance_by_time) + 1)
            (plot,) = ax2.plot(
                x_values,
                performance_by_time,
                # marker="o",
                label=line_label,
                color=color,
            )
            plot_legend.append(plot)
            ax2.set_xlim(-0.3, len(performance_by_time) + 0.3)

        ax2.locator_params(axis="y", nbins=30, tight=False)
        ax1.locator_params(axis="x", nbins=len(plotsets[0][0]) + 2, tight=False)
        ax2.tick_params(axis="y")

    ax1.legend(handles=plot_legend)
    ax1.margins(0.3, None)
    ax2.margins(0.3, None)

    plt.show()


##########################################################################################
# Trainer
##########################################################################################


class Phase(enum.Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATE = "validate"


class OptimizerKind(Enum):
    RMS_PROP = "rmsprop"
    ADAM_W = "adamw"
    ADAM = "adam"


def new_optimizer(
    kind: OptimizerKind,
    model_params: optimizer.ParamsT,
    learn_rate: float,
    weight_decay: float,
):
    match kind:
        case OptimizerKind.RMS_PROP:
            return torch.optim.RMSprop(
                model_params, lr=learn_rate, weight_decay=weight_decay
            )
        case OptimizerKind.ADAM_W:
            return torch.optim.AdamW(
                model_params, lr=learn_rate, weight_decay=weight_decay
            )
        case OptimizerKind.ADAM:
            return torch.optim.Adam(
                model_params, lr=learn_rate, weight_decay=weight_decay
            )


@dataclass
class TrainSettings:
    tabular_files: dict[str, str]
    image_directories: dict[str, str]
    target_features: list[str]
    device: torch.device
    rng_seed: int = 1
    run_name: str = "unnamed-trainrun"
    batch_size: int = 64
    num_epochs: int = 20
    test_per_epoch: bool = False
    checkpoint_per_epoch: bool = True
    optimizer: OptimizerKind = OptimizerKind.ADAM_W
    learn_rate: float = 1e-3
    weight_decay: float = 0.0001


class Trainer:
    def __init__(
        self,
        settings: TrainSettings,
        model: torch.nn.Module,
        train_loss_fn: torch.nn.Module,
        val_accuracy_fn,
        undo_transform_fn,
    ):
        self.settings = settings
        self.model = model
        self.optimizer = new_optimizer(
            kind=settings.optimizer,
            learn_rate=settings.learn_rate,
            weight_decay=settings.weight_decay,
            model_params=model.parameters(),
        )
        self.train_loss_fn = train_loss_fn
        self.val_accuracy_fn = val_accuracy_fn
        self.undo_transform_fn = undo_transform_fn
        self.train_start_time = int(time.time())

    def train(
        self, loaders: dict[str, data.DataLoader], num_epochs: int
    ) -> tuple[list[float], list[float]]:
        # Set network to training or evaluation mode. Model dropout layers are disabled,
        # and batch normalization uses entire population distribution during testing.
        self.model.to(self.settings.device)
        train_losses, validate_accuracies = [], []

        for epoch in range(1, num_epochs + 1):
            print(f"-- Epoch {epoch}/{num_epochs} --")
            # Execute one epoch of training.
            train_loss: float = self._train_epoch(
                loaders["train"], self.train_loss_fn, is_validation=False
            )
            train_losses.append(train_loss)
            print(f"TRAIN [ \033[34m Loss: {train_loss} \033[0m ]")

            # Run validation after training epoch.
            validate_acc: float = self._train_epoch(
                loaders["validate"], self.val_accuracy_fn, is_validation=True
            )
            validate_accuracies.append(validate_acc)
            print(f"VAL [ \033[33m Acc: {validate_acc} \033[0m ]")

            # Run test phase inferencing to "checkpoint" current progress.
            if self.settings.test_per_epoch:
                test_predictions: list[torch.Tensor] = self._test(
                    self.model, loaders["test"]
                )
                self._save_predictions(test_predictions, epoch)
            # Only checkpoint model and optimizer parameters if enabled or at last epoch.
            if self.settings.checkpoint_per_epoch or epoch >= num_epochs - 4:
                self._save_checkpoint(train_losses, validate_accuracies, epoch)

        # move parameters of the network back to CPU after training is complete.
        self.model.cpu()
        return train_losses, validate_accuracies

    def _train_epoch(
        self,
        loader: data.DataLoader,
        objective,
        is_validation: bool,
    ) -> float:
        self.model.eval() if is_validation else self.model.train()

        phase: str = "Validate" if is_validation else "Train"
        for batch in tqdm(loader, desc=f"{phase} Batches"):
            # Tensor, vector, vector (batch of images, feature vectors for each image,
            # 6 len target vectors for each image). Move each dataset batch to GPU.
            images, features, targets = [x.to(self.settings.device) for x in batch]
            features_no_id = features[:, 1:]

            with torch.set_grad_enabled(not is_validation):
                # Feed entire batch forward through network first, then backprop based on
                # loss over entire batch, recorded while we feed forward.
                predictions: torch.Tensor = self.model(images, features_no_id)
                batch_loss: torch.Tensor = objective.accumulate(predictions, targets)

                # TODO: remove validation flag since batch_loss could be none.
                if is_validation or batch_loss is None:
                    continue

                self.optimizer.zero_grad()
                batch_loss.backward()  # backprop and compute gradients of loss fn for weights
                # NOTE: Clip gradients to fix asymmetric gradient vector updates.
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()  # update weights by appliying gradients

        # Calculate the mean accuracy and loss over entire epoch (over all batches).
        return objective.compute()

    # TODO: Handle moving model if called directly.
    def _test(self, model: nn.Module, loader: data.DataLoader) -> list[torch.Tensor]:
        model.eval()
        test_predictions: list[torch.Tensor] = []

        # Execute one epoch of testing for each augmented variation of test dataset.
        for batch in tqdm(loader, desc="Testing Batches"):
            images, features, _ = [x.to(self.settings.device) for x in batch]
            feature_ids = features[:, 0].unsqueeze(1).int()
            features_no_id = features[:, 1:]

            with torch.no_grad():
                predictions: torch.Tensor = model(images, features_no_id).detach()
                predictions_with_id = (
                    torch.cat([feature_ids, predictions], dim=1).cpu().numpy()
                )

                if self.undo_transform_fn:
                    predictions_with_id[:, 1:] = self.undo_transform_fn(
                        predictions_with_id[:, 1:]
                    )

                test_predictions.append(predictions_with_id)

                # NOTE: Loss is not computed since there's no way to determine the loss over
                # each batch in testing because the test target regression values are not
                # provided in `test.csv`.

        # Returns the un-normalized model predictions over the current training batch.
        return test_predictions

    def test_past_epoch(
        self, model: nn.Module, loader: data.DataLoader, past_epoch: int
    ) -> None:
        # Run test predictions over the model weights from the epoch with best validation accuracy.
        file_name: str = self._new_file_name("trainstate", past_epoch, "pth")
        saved_train_state = torch.load(file_name, map_location=self.settings.device)

        model.load_state_dict(saved_train_state["state_dict"], strict=True)
        model.to(self.settings.device)

        past_test_predictions = self._test(model, loader)
        self._save_predictions(past_test_predictions, past_epoch)
        model.cpu()

    def manual_test_past_epoch(
        self, model: nn.Module, loader: data.DataLoader, file_name: str
    ) -> None:
        saved_train_state = torch.load(file_name, map_location=self.settings.device)
        model.load_state_dict(saved_train_state["state_dict"], strict=True)
        model.to(self.settings.device)

        self._manual_save_predictions(
            self._test(model, loader), file_name + "-predictions.csv"
        )
        model.cpu()

    def _manual_save_predictions(
        self, predictions: list[torch.Tensor], file_name: str
    ) -> None:
        target_columns: list[str] = ["id"] + self.settings.target_features

        with open(file_name, "w") as file:
            writer = csv.writer(file)
            writer.writerow(target_columns)
            for prediction in predictions:
                writer.writerows(prediction)

    # TODO: Refactor to call _manual_save_predictions.
    def _save_predictions(self, predictions: list[torch.Tensor], epoch: int) -> None:
        file_name: str = self._new_file_name("testpredict", epoch, "csv")
        target_columns: list[str] = ["id"] + self.settings.target_features

        with open(file_name, "w") as file:
            writer = csv.writer(file)
            writer.writerow(target_columns)
            for prediction in predictions:
                writer.writerows(prediction)

    def _save_checkpoint(
        self, train_losses: list[float], validate_accuracies: list[float], epoch: int
    ) -> None:
        file_name: str = self._new_file_name("trainstate", epoch, "pth")
        train_state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_losses": train_losses,
            "validate_accuracies": validate_accuracies,
        }
        torch.save(train_state, file_name)

    def _new_file_name(self, content_kind: str, epoch: int, extension: str):
        return (
            f"{self.settings.run_name}-{self.train_start_time}"
            f"-{content_kind}-epoch{epoch:04d}.{extension}"
        )


##########################################################################################
# Loss Functions
##########################################################################################


class BatchMSLoss:
    def __init__(self):
        super(BatchMSLoss, self).__init__()
        self.loss_fn = nn.MSELoss()
        self._reset_next_epoch()

    def _reset_next_epoch(self) -> None:
        self.sum_batch_losses: float = 0
        self.num_datapoints: int = 0

    def accumulate(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        batch_size: int = len(targets)
        batch_loss: torch.Tensor = self.loss_fn(predictions, targets)

        # Update terms to eventually compute mean loss over all batches in current epoch.
        self.sum_batch_losses += batch_loss.item() * batch_size
        self.num_datapoints += batch_size

        return batch_loss

    def compute(self) -> float:
        if self.num_datapoints == 0:
            return 0
        mean_epoch_loss = self.sum_batch_losses / self.num_datapoints
        self._reset_next_epoch()
        return mean_epoch_loss


class BatchR2Score:
    def __init__(self, device: torch.device, mean_target: torch.Tensor):
        super(BatchR2Score, self).__init__()
        self.device = device
        self.mean_target: torch.Tensor = mean_target.to(device)
        self._reset_next_epoch()

    def _reset_next_epoch(self) -> None:
        # vectors
        self.sum_residual_squares = torch.zeros((1, 6), device=self.device)
        self.sum_total_squares = torch.zeros((1, 6), device=self.device)
        self.epoch_predictions = torch.tensor([], device=self.device)
        self.epoch_targets = torch.tensor([], device=self.device)

    def accumulate(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        # Add the squared residual error for every target value vs predicted value, for
        # all datapoints over current batch.

        # Subtracts each target vector (row) by each prediction vector via broadcasting.
        batch_residuals: torch.Tensor = targets - predictions
        # Square each vector, same as squaring element wise over entire tensor. Then sum
        # over each target batch residual squared vector (row) in tensor.

        self.sum_residual_squares += batch_residuals.square().sum(dim=0)

        # Add the total squares for every target value vs predicted value, for all datapoints
        # over current batch.
        #
        batch_totals: torch.Tensor = targets - self.mean_target
        self.sum_total_squares += batch_totals.square().sum(dim=0)

        self.epoch_predictions = torch.cat([self.epoch_predictions, predictions], dim=0)
        self.epoch_targets = torch.cat([self.epoch_targets, targets], dim=0)

    def compute(self) -> float:
        real_r2_score = torchmetrics.functional.r2_score(
            self.epoch_predictions, self.epoch_targets
        ).item()

        # if self.sum_total_squares == 0:
        #     return 0

        # Compute the R² score for each trait by performing element-wise division, obtains
        # multivariate R² score vector for all output traits.
        # https://pytorch.org/docs/stable/generated/torch.div.html
        r2_scores = 1 - (self.sum_residual_squares / self.sum_total_squares)
        mean_r2_score = r2_scores.mean().item()

        print(f"Batch-wise Mean R² Score: {mean_r2_score}")
        print(f"Torchmetrics R² Score: {real_r2_score}")

        self._reset_next_epoch()
        # Take average, could squeeze it to a vector but no point.
        return mean_r2_score


##########################################################################################
# Datasets
##########################################################################################


class SquareRootTransform:
    """
    Custom transformation that applies square root normalization and standard scaling
    to a tabular dataset.

    The square root normalization transforms each element in the dataset by applying
    the formula:
    ```
        x' = sign(x) * sqrt(|x|)
    ```
    where sign(x) is the sign of x and |x| is the absolute value of x.

    After applying the square root normalization, the data is then standardized to have
    mean 0 and standard deviation 1 using a StandardScaler. This ensures that the feature
    columns are scaled such that they are centered around zero with unit variance.

    The inverse transformation reverts the standardized data back to the original scale
    following the reverse of the square root normalization.
    """

    def __init__(self, fit_on: np.ndarray, outlier_zscore: float):
        self.outlier_zscore = outlier_zscore
        self.X_train_means = fit_on.mean(axis=0)
        self.X_train_stdevs = fit_on.std(axis=0)

        self.X_train_sqrt_means, self.X_train_sqrt_stdevs = self._compute_tf_mean_std(
            fit_on
        )

    def _compute_tf_mean_std(self, X_train: np.ndarray) -> tuple:
        # Calculate boundaries for outlier detection for each column
        X_min_bound = self.X_train_means - self.outlier_zscore * self.X_train_stdevs
        X_max_bound = self.X_train_means + self.outlier_zscore * self.X_train_stdevs

        # Impute outliers with the mean of non-outlier data for each column
        X_imputed = np.where(
            (X_train < X_min_bound) | (X_train > X_max_bound),
            self.X_train_means,
            X_train,
        )
        X_imputed_stabilized = np.sign(X_imputed) * np.sqrt(np.absolute(X_imputed))

        return np.mean(X_imputed_stabilized, axis=0), np.std(
            X_imputed_stabilized, axis=0
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Can process entire dataset, or batch at a time.
        """
        num_cols = X.shape[1]
        X_train_means, X_train_stdevs = (
            self.X_train_means[:num_cols],  # end is not included...
            self.X_train_stdevs[:num_cols],
        )
        X_train_sqrt_means, X_train_sqrt_stdevs = (
            self.X_train_sqrt_means[:num_cols],
            self.X_train_sqrt_stdevs[:num_cols],
        )

        # for each column
        #
        # if the training(column) skewness is above threshold:
        #   apply sqrt transform to fix distribution
        # apply standardization

        # Calculate boundaries for outlier detection for each column
        X_min_bound = X_train_means - 3 * X_train_stdevs
        X_max_bound = X_train_means + 3 * X_train_stdevs

        # If test col feature value is outlier, take the the computed training mean instead.
        X_imputed = np.where((X < X_min_bound) | (X > X_max_bound), X_train_means, X)
        X_transformed = np.sign(X_imputed) * np.sqrt(np.absolute(X_imputed))

        # If the training column feature was skewed above threshold, stabilize the
        # test feature column's distribution.
        return (X_transformed - X_train_sqrt_means) / X_train_sqrt_stdevs

    def undo(self, X_applied: np.ndarray) -> np.ndarray:
        num_cols = X_applied.shape[1]
        # TODO: Checking if data to un-normalize is the input features or predictions
        # based on number of columns. This is a very bad way to do this, fix it.
        is_predictions: bool = num_cols == 6
        # TODO: are these indices correct?
        start, end = (163, 170) if is_predictions else (0, num_cols)

        X_train_sqrt_means, X_train_sqrt_stdevs = (
            self.X_train_sqrt_means[start:end],
            self.X_train_sqrt_stdevs[start:end],
        )

        X_transformed = (X_applied * X_train_sqrt_stdevs) + X_train_sqrt_means
        X_imputed = np.sign(X_transformed) * (np.abs(X_transformed) ** 2)

        return X_imputed


from scipy import stats


# just instantiate this for target labels...
class HybridTransform:
    def __init__(self, fit_on: np.ndarray, eligible_cols: list[bool] = None):
        assert eligible_cols is None or fit_on.shape[1] == len(
            eligible_cols
        ), "The length of 'fit_on' and 'eligible_cols' must be the same."

        self.eligible_cols = eligible_cols
        self.num_cols = fit_on.shape[1]
        x_train = fit_on.copy()

        self.og_X_train_means = fit_on.mean(axis=0)
        self.og_X_train_stdevs = fit_on.std(axis=0)

        self.min_values = x_train.min(axis=0)
        x_train -= self.min_values

        x_train_skew = stats.skew(x_train, axis=0)
        is_mildly_skewed = (x_train_skew >= 0.5) & (x_train_skew < 1.0)
        is_highly_skewed = x_train_skew >= 1.0

        self.mild_skew_indices = np.where(is_mildly_skewed)[0]
        self.high_skew_indices = np.where(is_highly_skewed)[0]

        x_train[:, self.high_skew_indices] = np.log10(
            x_train[:, self.high_skew_indices] + 1e-6
        )
        x_train[:, self.mild_skew_indices] = np.sqrt(x_train[:, self.mild_skew_indices])

        self.normalized_means = np.mean(x_train, axis=0)
        self.normalized_stds = np.std(x_train, axis=0)

        self.p01_outliers = np.percentile(x_train, 0.01, axis=0)
        self.filtered_indices = np.all(x_train > self.p01_outliers, axis=1)

    # TODO: store a column mask... should only transform on last 6 target columns
    def transform(self, X_: np.ndarray, content: str):
        # NOTE: we don't know the num of cols in X, but only care about the last 6.
        # (or generally last k depending on length of dataset we fit onto)
        if content == "features":
            start, end = 1, 164
        if content == "targets":
            start, end = 164, 170

        X = X_[:, start:end]

        # print("TOTAL NUM COLUMNS: ", self.num_cols)
        # print("TOTAL INPUT NUM COLUMNS: ", self.num_cols)

        for col in range(0, self.num_cols):
            if self.eligible_cols is None or self.eligible_cols[col] is True:
                X[:, col] -= self.min_values[col]
                if col in self.high_skew_indices:
                    X[:, col] = np.log10(X[:, col] + 1e-6)
                if col in self.mild_skew_indices:
                    X[:, col] = np.sqrt(X[:, col])

                # if this becomes negative, that is ok since it could be left or right of mean 0.
                X[:, col] = (
                    X[:, col] - self.normalized_means[col]
                ) / self.normalized_stds[col]
            else:
                X[:, col] = (
                    X[:, col] - self.og_X_train_means[col]
                ) / self.og_X_train_stdevs[col]

        # Only remove outlier feature vectors from training dataset.
        if X_.shape[0] == self.filtered_indices.shape[0]:
            print("PRUNING!!!!!!!!!!")
            return X_[self.filtered_indices]
        return X_

    def undo_transform(self, y: np.ndarray):
        assert y.shape[1] == self.num_cols
        y = y.copy()

        # Select only the last 6 elements from normalized_stds and normalized_means
        # assuming this only un-transforms predictions.
        y = (y * self.normalized_stds) + self.normalized_means

        # Reverse skewness-based transformations
        y[:, self.high_skew_indices] = np.power(10, y[:, self.high_skew_indices]) - 1e-6
        y[:, self.mild_skew_indices] = np.square(y[:, self.mild_skew_indices])

        # Add the minimum feature values per each column back.
        y += self.min_values
        return y


class MergedImageDataset(data.Dataset):
    def __init__(
        self,
        image_dir: str,
        tabular_csv_path: str,
        target_features: list[str] = [],
        image_transform=None,
        feature_transform=None,
        target_transform=None,
    ):
        tabular_header: pd.DataFrame = pd.read_csv(tabular_csv_path, nrows=0)
        feature_datatypes = {col: "float64" for col in tabular_header.columns}
        feature_datatypes["id"] = "int"

        tabular_records: pd.DataFrame = pd.read_csv(
            tabular_csv_path, dtype=feature_datatypes
        )

        # X = tabular_records.iloc[:, 1:164].values

        if feature_transform:
            print(">>>>>>>> TRANSFORMING FEATURES")
            # Transform all image features and labels on training feature vectors, if
            # they exist after index 164. ID column is excluded from transformation.
            transformed = feature_transform(tabular_records.values, "features")
            tabular_records = pd.DataFrame(transformed, columns=tabular_records.columns)
        if target_transform:
            print(">>>>>>>> TRANSFORMING TARGETS")
            transformed = target_transform(tabular_records.values, "targets")
            tabular_records = pd.DataFrame(transformed, columns=tabular_records.columns)

        self.tabular_records = tabular_records
        self.image_transform = image_transform
        self.image_dir = image_dir
        self.target_features = target_features if "train" in image_dir else []

    def __len__(self):
        return len(self.tabular_records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load single image from folder.
        image_filename = str(int(self.tabular_records.iloc[idx, 0])) + ".jpeg"
        # Obtains image file name from id column of each image feature vector.
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        # Load image feature vector from CSV file (excluding the first column).
        # Keeps each feature vector (image) ID for Kaggle submission.
        image_features = self.tabular_records.iloc[idx, :164].to_numpy()
        image_features = torch.tensor(image_features, dtype=torch.double)

        # Load image target values vector from CSV file.
        target_col_idxs = [
            self.tabular_records.columns.get_loc(feature_name + "_mean")
            for feature_name in self.target_features
        ]

        image_targets = self.tabular_records.iloc[idx, target_col_idxs].to_numpy()
        image_targets = torch.tensor(image_targets, dtype=torch.double)

        if self.image_transform:
            image = self.image_transform(image)

        return image, image_features, image_targets


def load_dataset(
    settings: TrainSettings,
    phase: Phase,
    image_transform: transforms.Compose = None,
    feature_transform=None,
    target_transform=None,
) -> data.DataLoader:
    dataset = MergedImageDataset(
        settings.image_directories[phase.value],
        settings.tabular_files[phase.value],
        settings.target_features,
        image_transform,
        feature_transform,
        target_transform,
    )

    print("DONE LOADING ", phase)

    is_shuffled: bool = True if phase == Phase.TRAIN else False
    return data.DataLoader(dataset, settings.batch_size, shuffle=is_shuffled)
