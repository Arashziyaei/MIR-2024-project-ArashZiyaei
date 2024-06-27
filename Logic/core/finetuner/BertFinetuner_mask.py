import json
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        # TODO: Implement initialization logic
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.top_genres = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=top_n_genres)
        self.trainer = None
        self.data = None
        self.df = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        # TODO: Implement dataset loading logic
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        self.df = pd.DataFrame(self.data)

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        # TODO: Implement genre filtering and visualization logic
        genres = []
        for item in self.df['genres']:
            if item is not None:
                for genre in item:
                    genres.append(genre)
        genre_counts = Counter(genres)

        self.top_genres = [genre for genre, count in genre_counts.most_common(self.top_n_genres)]

        filtered_df = self.df[self.df['genres'].apply(lambda genres: any(genre in self.top_genres for genre in genres))]
        filtered_df['genres'] = filtered_df['genres'].apply(
            lambda genres: [genre for genre in genres if genre in self.top_genres])
        self.df = filtered_df

        plt.figure(figsize=(10, 5))
        sns.barplot(x=self.top_genres, y=[count for genre, count in genre_counts.most_common(self.top_n_genres)])
        plt.title('top genres distribution')
        plt.xlabel('genres')
        plt.ylabel('count')
        plt.show()

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        # TODO: Implement dataset splitting logic
        X = self.df['first_page_summary']
        y = LabelEncoder().fit_transform(self.df['genres'].str[0].tolist())

        self.x_train, X_temp, self.y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
        self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(X_temp, y_temp, test_size=val_size,
                                                                            random_state=42)

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        # TODO: Implement dataset creation logic
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        # TODO: Implement BERT fine-tuning logic
        train_encodings = self.tokenizer(list(map(str, self.x_train)), truncation=True, padding=True)
        val_encodings = self.tokenizer(list(map(str, self.x_val)), truncation=True, padding=True)

        train_dataset = self.create_dataset(train_encodings, self.y_train)

        val_dataset = self.create_dataset(val_encodings, self.y_val)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # TODO: Implement metric computation logic
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        # TODO: Implement model evaluation logic
        test_encodings = self.tokenizer(list(map(str, self.x_test)), truncation=True, padding=True)
        test_dataset = self.create_dataset(test_encodings, self.y_test)

        pred, _, _ = self.trainer.predict(test_dataset)
        preds = pred.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, preds, average='weighted')
        acc = accuracy_score(self.y_test, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        # TODO: Implement model saving logic
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)


class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        # TODO: Implement initialization logic
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        # TODO: Implement item retrieval logic
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        # TODO: Implement length computation logic
        return len(self.labels)