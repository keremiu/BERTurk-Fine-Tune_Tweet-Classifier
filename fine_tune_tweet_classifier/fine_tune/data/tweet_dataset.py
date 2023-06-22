import torch

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, tokens: list[dict], labels: list[str], texts: list[str]):
        self.input_ids = list( map( lambda x: x["input_ids"], tokens ) ) 
        self.attention_masks = list( map( lambda x: x["attention_mask"], tokens ) )

        self.num_classes = len(set(labels))
        self.labels = labels
        
        self.texts = texts
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        one_hot_encoding = torch.zeros(self.num_classes)
        one_hot_encoding[self.labels[index]] = 1

        return torch.tensor(self.input_ids[index]), torch.tensor(self.attention_masks[index]), one_hot_encoding, self.texts[index]