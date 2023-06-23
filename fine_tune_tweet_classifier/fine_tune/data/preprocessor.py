from .tweet_dataset import TweetDataset

import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
import transformers
import pandas
import re

class Preprocessor():
    def __init__(self, tokenizer: transformers.BertTokenizer):
        self.tokenizer = tokenizer
        self.label_encoder = LabelEncoder()

    def encode_labels(self, labels: pandas.Series):
        encoded_labels = self.label_encoder.fit_transform(labels)

        encoded_labels_mapping = {label: encode for encode, label in enumerate(self.label_encoder.classes_)}
        logging.info(f"Encoded Labels: {encoded_labels_mapping}")
        print(f"Encoded Labels: {encoded_labels_mapping}")

        return encoded_labels

    def reverse_encoded_label(self, argument: list):
        return self.label_encoder.inverse_transform(argument)

    def remove_emojis(self, series_text: pandas.Series):
        emoji_pattern = re.compile(
             "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
             "]+", 
            flags=re.UNICODE
        )

        return series_text.apply(lambda text: emoji_pattern.sub(r'', text))

    def prepare_inputs(self, labeled_tweets: pandas.DataFrame, validation_size: float, batch_size: int):
        # Encode labels
        labeled_tweets["label"] = self.encode_labels(labeled_tweets["label"])

        # Remove emojis
        labeled_tweets["text"] = self.remove_emojis(labeled_tweets["text"])

        # Remove mentions, hashtags and links
        filter_func = lambda word: not word.startswith("@") and not word.startswith("#") and not word.startswith("http")     
        labeled_tweets["text"] = labeled_tweets["text"].apply(  # 1. For every row 
            lambda row: " ".join(                               # 4. Join the unfiltered words into a string
                filter(
                    filter_func,                                # 3. If a word starts with '@' or "#" or 'http', filter it out
                    row.split()                                 # 2. Split the text into a list of words
                )
            )
        )

        # Tokenize texts
        max_length = labeled_tweets.text.str.len().max() if labeled_tweets.text.str.len().max() < 512 else 512
        labeled_tweets["tokens"] = labeled_tweets["text"].apply(
            lambda text: self.tokenizer(text, padding="max_length", max_length=max_length, truncation=True)
        )   

        # Split the data into train and test
        train_tweets, validation_tweets = train_test_split(labeled_tweets, shuffle=True, test_size=validation_size)

        # Initialize datasets
        train_dataset = TweetDataset(train_tweets["tokens"], train_tweets["label"].to_list(), train_tweets["text"].to_list())
        validation_dataset = TweetDataset(validation_tweets["tokens"], validation_tweets["label"].to_list(), validation_tweets["text"].to_list())

        # Initialize dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=list(set(train_dataset.labels)),
            y=train_dataset.labels
        )

        return train_dataloader, validation_dataloader, class_weights