from .tweet_dataset import TweetDataset

import logging

from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader
import transformers
import pandas
import numpy
import re

class Preprocessor():
    def __init__(self, tokenizer: transformers.BertTokenizer):
        self.tokenizer = tokenizer
        self.encoding_indices: dict

    def encode_labels(self, labels: pandas.Series, return_class_weights: bool):
        unique_labels = labels.apply(
            lambda x: re.sub(r'[ \'\[\]]', '', x).split(",")    # Labels are converted to list objects from strings
        ).explode().unique()                                    # Then the unique elements in all of those lists are obtained

        self.encoding_indices = {index: label for index, label in enumerate(unique_labels)}

        df_one_hot = pandas.DataFrame(0, index=numpy.arange(len(labels)), columns=unique_labels)
        df_one_hot["label"] = labels
        
        class_weights = []
        num_entries = len(df_one_hot)
        for unique_label in unique_labels:
            df_one_hot.loc[df_one_hot.label.str.contains(unique_label), unique_label] = 1  

            pos_weight = sum(df_one_hot[unique_label] == 1) / num_entries
            class_weights.append(pos_weight)

        if return_class_weights == True:
            return df_one_hot[unique_labels].values.tolist(), class_weights
        else:
            return df_one_hot[unique_labels].values.tolist()

    def decode_labels(self, argument: list):
        decoded_labels = [self.encoding_indices[index] for index, value in enumerate(argument) if value == 1]
        
        return decoded_labels

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
        labeled_tweets["label"], class_weights = self.encode_labels(labeled_tweets["label"], return_class_weights=True)

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

        return train_dataloader, validation_dataloader, class_weights