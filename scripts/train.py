#!/usr/bin/env python3.11

"""
Training script for fine-tuning a DistilBERT model to classify emotions in text data.
"""

import os
import logging
import typing
import pickle
import yaml

from datasets import load_dataset, DatasetDict
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# TODO: Not sure about BatchEncoding
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    BatchEncoding,
)
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EmotionClassifierTrainer:
    """Class that encapsulates the variables and functions necessary for fine tuning the DistelBERT model to classify text by emotions."""

    def __init__(self, config_path: str) -> None:
        """
        Initialize the EmotionClassifierTrainer with configuration.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.dataset_folder_path = self.config["dataset_folder_path"]
        self.label_encoder_path = self.config["label_encoder_path"]
        self.model_name = self.config["model"]
        self.num_labels = self.config["num_labels"]
        self.batch_size = self.config["batch_size"]
        self.model_save_path = self.config["model_save_path"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tf_model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels
        )
        self.label_encoder = LabelEncoder()

    # TODO: Not sure about the type hints!
    def tokenize(self, data: typing.Dict[str, typing.Any]) -> BatchEncoding:
        """
        Tokenize and truncate the input text data.

        Args:
            data (dict): Input data containing text to be tokenized.

        Returns:
            BatchEncoding: Tokenized and truncated data.
        """
        return self.tokenizer(
            data["text"], padding=True, truncation=True, return_tensors="tf"
        )

    @staticmethod
    def extract_labels(features: typing.Any, labels: typing.Any) -> typing.Any:
        """
        Extract true labels from the dataset.

        Args:
            features (Any): Input features.
            labels (Any): True labels.

        Returns:
            Any: True labels.
        """
        return labels

    @staticmethod
    def evaluate_model(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> typing.Dict[str, float]:
        """
        Evaluate the model using accuracy and F1 score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            dict: Dictionary containing accuracy and F1 score.
        """
        f1 = f1_score(y_true, y_pred, average="weighted")
        acc = accuracy_score(y_true, y_pred)
        return {"accuracy": acc, "f1": f1}

    # TODO: Not sure about the typing!
    def load_all_datasets(self) -> typing.Tuple[DatasetDict, DatasetDict, DatasetDict]:
        """
        Load all datasets from the dataset folder path.

        Returns:
            tuple: Training, validation, and test datasets.
        """
        datasets_dict = {}
        for dirname, _, filenames in os.walk(self.dataset_folder_path):
            for filename in filenames:
                filename_title = filename.split(".txt")[0]
                filename_path = os.path.join(dirname, filename)
                datasets_dict[filename_title] = filename_path

        emotions_train = load_dataset(
            "csv",
            data_files=datasets_dict["train"],
            delimiter=";",
            names=["text", "label"],
        )
        emotions_val = load_dataset(
            "csv",
            data_files=datasets_dict["val"],
            delimiter=";",
            names=["text", "label"],
        )
        emotions_test = load_dataset(
            "csv",
            data_files=datasets_dict["test"],
            delimiter=";",
            names=["text", "label"],
        )

        return emotions_train, emotions_val, emotions_test

    # TODO: Not sure about the typing!
    def encode_labels_for_tf_model(
        self, emotions_train_encoded: DatasetDict, emotions_val_encoded: DatasetDict
    ) -> typing.Tuple[DatasetDict, DatasetDict]:
        """
        Encode labels for TensorFlow model and save the label encoder.

        Args:
            emotions_train_encoded (DatasetDict): Encoded training dataset.
            emotions_val_encoded (DatasetDict): Encoded validation dataset.

        Returns:
            tuple: Encoded training and validation datasets.
        """
        labels_train = emotions_train_encoded["train"]["label"]
        labels_val = emotions_val_encoded["train"]["label"]

        label_encoded_train = self.label_encoder.fit_transform(labels_train)
        emotions_train_encoded["train"] = emotions_train_encoded["train"].add_column(
            "encoded_label", label_encoded_train
        )

        label_encoded_val = self.label_encoder.transform(labels_val)
        emotions_val_encoded["train"] = emotions_val_encoded["train"].add_column(
            "encoded_label", label_encoded_val
        )

        with open(self.label_encoder_path, "wb") as file:
            pickle.dump(self.label_encoder, file)

        return emotions_train_encoded, emotions_val_encoded

    @staticmethod
    def confusion_matrix_plot(
        y_pred: np.ndarray, y_true: np.ndarray, labels: typing.List[str], clf: str
    ) -> None:
        """
        Create and display confusion matrix.

        Args:
            y_pred (np.ndarray): Predicted labels.
            y_true (np.ndarray): True labels.
            labels (list): List of labels.
            clf (str): Classifier name.
        """
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        _, ax = plt.subplots(figsize=(6, 6))
        confm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        confm.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
        plt.title(f"Normalized confusion matrix {clf}")
        plt.show()

    def train(self) -> None:
        """
        Train the emotion classifier model.
        """
        try:
            emotions_train, emotions_val, _ = self.load_all_datasets()

            # Tokenize
            emotions_train_encoded = emotions_train.map(
                self.tokenize, batched=True, batch_size=None
            )
            emotions_val_encoded = emotions_val.map(
                self.tokenize, batched=True, batch_size=None
            )

            emotions_train_encoded, emotions_val_encoded = (
                self.encode_labels_for_tf_model(
                    emotions_train_encoded, emotions_val_encoded
                )
            )

            tokenizer_columns = self.tokenizer.model_input_names

            # Fine Tune
            tf_train = emotions_train_encoded["train"].to_tf_dataset(
                columns=tokenizer_columns,
                label_cols=["encoded_label"],
                shuffle=True,
                batch_size=self.batch_size,
            )

            tf_val = emotions_val_encoded["train"].to_tf_dataset(
                columns=tokenizer_columns,
                label_cols=["encoded_label"],
                shuffle=False,
                batch_size=self.batch_size,
            )

            self.tf_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )

            self.tf_model.fit(tf_train, validation_data=tf_val, epochs=3)
            y_pred_tf = self.tf_model.predict(tf_val)

            # Process Predicted Labels from Model
            logits = y_pred_tf.logits
            y_pred = np.argmax(logits, axis=1)

            y_true = tf_val.map(self.extract_labels)
            y_true = tf.concat(list(y_true.as_numpy_iterator()), axis=0)

            scores = self.evaluate_model(y_true, y_pred)
            logging.info(f"Evaluation Scores: {scores}")

            labels = np.unique(emotions_train["train"]["label"])
            self.confusion_matrix_plot(y_pred, y_true, labels, "TF Model")

            # Save Model and Tokenizer
            self.tf_model.save_pretrained(self.model_save_path)
            self.tokenizer.save_pretrained(self.model_save_path)
            logging.info(f"Model and tokenizer saved to {self.model_save_path}")

        except Exception as e:
            logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    trainer = EmotionClassifierTrainer(config_path="config.yaml")
    trainer.train()
