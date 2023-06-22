from src.services.config_service import ConfigService
from src.services.data_service import DataService

from .model.bert_model import BERTModel
from .data.preprocessor import Preprocessor

import logging

from sklearn.metrics import f1_score, confusion_matrix
import transformers
import torch

class Trainer():
    def __init__(self, config_service: ConfigService, data_service: DataService):
        self.config_service = config_service
        self.data_service = data_service
        
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.config_service.bert_model_name)
        
        self.preprocessor = Preprocessor(self.tokenizer)        

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bert = transformers.BertModel.from_pretrained(self.config_service.bert_model_name)
        self.model = BERTModel(bert, self.config_service.layers).to(self.device)

    def test(self):
        labeled_tweets = self.data_service.read_test_tweets()
        
        test_dataloader, _ = self.preprocessor.prepare_inputs(
            labeled_tweets=labeled_tweets, 
            validation_size=1, 
            batch_size=1
        )

        self.model.eval()
        for test_input_ids, test_attention_masks, test_labels, test_texts in test_dataloader:
            test_output = self.model(test_input_ids.to(self.device), test_attention_masks.to(self.device))

            test_f1 = f1_score(test_labels, test_output)

    def train(self):
        labeled_tweets = self.data_service.read_training_tweets()

        train_dataloader, validation_dataloader = self.preprocessor.prepare_inputs(
            labeled_tweets=labeled_tweets, 
            validation_size=self.config_service.validation_size, 
            batch_size=self.config_service.batch_size
        )

        loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config_service.learning_rate)

        if self.config_service.lr_scheduler != None:
            ...

        for epoch in range(1, self.config_service.num_epochs+1):
            training_f1 = 0
            training_loss = 0

            self.model.train()
            for input_ids, attention_masks, labels, _ in train_dataloader:
                output = self.model(input_ids.to(self.device), attention_masks.to(self.device))

                loss = loss_fn(output, labels.to(self.device))
                training_loss += loss.item()

                training_f1 += f1_score(labels, output, average="macro")

                self.model.zero_grad()
                loss.backwards()
                optimizer.step()


            validation_f1 = 0
            validation_loss = 0

            self.model.eval()
            for val_input_ids, val_attention_masks, val_labels, _ in validation_dataloader:
                val_output = self.model(val_input_ids.to(self.devices), val_attention_masks.to(self.device))

                val_loss = loss_fn(val_output, val_labels.to(self.debice))
                validation_loss += val_loss

                validation_f1 += f1_score(val_labels, val_output)

            
            info = f"Epoch: {epoch}\t|\t" \
                   f"Validation F1 Score: {validation_f1/len(validation_dataloader)}\t|\tValidation Loss: {validation_loss/len(validation_dataloader)}\t|\t" \
                   f"Training F1 Score: {training_f1/len(train_dataloader)}\t|\tTraining Loss: {training_loss/len(train_dataloader)}"
            
            logging.info(info)
            print(info)
