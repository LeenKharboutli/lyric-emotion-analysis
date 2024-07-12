"""
Load fine tuned distilBERT model and use it to classify lyrics by emotion.
Save the model output as CSV.
"""
import typing
import logging
import pickle
import yaml
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LyricsEmotionClassifier:
    """Class that encapsulates the global variables and functions
       necessary to run the lyrics classification.
    """
    def __init__(self, config_path: str) -> None:
        """
        Initialize the LyricsEmotionClassifier with configuration.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        self.num_labels = config["num_labels"]
        self.batch_size = config["batch_size"]
        self.model_save_path = config["model_save_path"]
        self.label_encoder_path = config["label_encoder_path"]
        self.raw_lyrics_dataset_path = config["raw_lyrics_dataset_path"]
        self.predicted_lyrics_classification_dataset_path = config[
            "predicted_lyrics_classification_dataset_path"
        ]

        self.tf_model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_save_path, num_labels=self.num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_save_path)

        with open(self.label_encoder_path, "rb") as file:
            le = pickle.load(file)
        self.encoding_dict = dict(zip(le.transform(le.classes_), le.classes_))

    # def tokenize_truncate(self, data: typing.Dict[str, typing.Any]) -> BatchEncoding:
    def tokenize_truncate(self, data):
        """
        Truncate if token length exceeds max_length.

        Args:
            data (dict): Input data containing text to be tokenized.

        Returns:
            transformers.models.distilbert.tokenization_distilbert_fast.DistilBertTokenizerFast:
                configured tokenizer.
        """
        max_length = 512
        return self.tokenizer(
            data["text"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="tf",
        )

    #TODO: not sure about the type hints!
    def get_predicted_labels(self, lyrics_y_pred_tf: typing.Any) -> np.ndarray:
        """
        Get the predicted labels from the model output.

        Args:
            lyrics_y_pred_tf: The output logits from the model prediction.

        Returns:
            np.ndarray: Array of predicted labels.
        """
        lyrics_logits = lyrics_y_pred_tf.logits
        lyrics_y_pred = np.argmax(lyrics_logits, axis=1)
        logging.info(f"Predicted Labels: {lyrics_y_pred}")
        return lyrics_y_pred

    #TODO: not sure about the type hints!
    def save_data(self, lyrics_test_encoded: typing.Any, lyrics_y_pred: np.ndarray) -> None:
        """
        Save the predicted data to a CSV file.

        Args:
            lyrics_test_encoded: Encoded test data containing lyrics text.
            lyrics_y_pred (np.ndarray): Array of predicted labels.

        Returns:
            None
        """
        lyrics_text = lyrics_test_encoded["train"]["text"]
        predicted_lyrics_labels = [self.encoding_dict[val] for val in lyrics_y_pred]
        lyrics_df = pd.DataFrame(
            {
                "lyrics_text": lyrics_text,
                "predicted_emotion_label": predicted_lyrics_labels,
                "predicted_encoded_label": lyrics_y_pred,
            }
        )
        lyrics_df.to_csv(self.predicted_lyrics_classification_dataset_path, index=False)
        logging.info(f"Data saved to {self.predicted_lyrics_classification_dataset_path}")

    def run(self) -> None:
        """
        Run the classification process.
        """
        try:
            lyrics_test_data = load_dataset(
                "csv", data_files=self.raw_lyrics_dataset_path, delimiter=",", names=["text"]
            )  # Might have to amend the delimiter, based on how we write the data

            lyrics_test_encoded = lyrics_test_data.map(
                self.tokenize_truncate, batched=True, batch_size=None
            )

            tokenizer_columns = self.tokenizer.model_input_names
            tf_lyrics = lyrics_test_encoded["train"].to_tf_dataset(
                columns=tokenizer_columns,  # label_cols=["encoded_label"],
                shuffle=False,
                batch_size=self.batch_size,
            )
            lyrics_y_pred_tf = self.tf_model.predict(tf_lyrics)

            lyrics_y_pred = self.get_predicted_labels(lyrics_y_pred_tf)

            # Display some examples
            for i, _ in enumerate(lyrics_y_pred):

                if i > 10:
                    break

                predicted_label_index = lyrics_y_pred[i]
                predicted_label = self.encoding_dict[predicted_label_index]
                logging.info(
                    f"Predicted Label: {predicted_label}\nText: {lyrics_test_encoded['train']['text'][i]}\n\n"
                )

            self.save_data(lyrics_test_encoded, lyrics_y_pred)
        except Exception as e:
            logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    classifier = LyricsEmotionClassifier(config_path="config.yaml")
    classifier.run()
