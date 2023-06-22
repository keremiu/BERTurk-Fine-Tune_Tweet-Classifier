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

        self.device = torch.device(self.config_service.device)

        bert = transformers.BertModel.from_pretrained(self.config_service.bert_model_name)
        self.model = BERTModel(bert, self.config_service.layers).to(self.device)

    def test(self):
        labeled_tweets = self.data_service.read_test_tweets()
        
        test_dataloader, _ = self.preprocessor.prepare_inputs(
            labeled_tweets=labeled_tweets, 
            validation_size=1, 
            batch_size=1
        )

        labels = []
        preds = []

        self.model.eval()
        for test_input_ids, test_attention_masks, test_label, test_text in test_dataloader:
            test_output = self.model(test_input_ids.to(self.device), test_attention_masks.to(self.device))

            labels.append(test_label.argmax().item())
            preds.append(test_output.argmax().item())

            info = f"Text: {test_text}\nLabel: {test_label}\nOutput: {test_output.argmax(axis=1)}\n"
            print(info)
            logging.info(info)

        end_info = f"Test F1 Score: {f1_score(labels, preds, average='macro')}\nTest Confusion Matrix: {confusion_matrix(labels, preds)}"
        print(end_info)
        logging.info(end_info)        


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
            training_loss = 0
            training_preds = []
            training_labels = []

            self.model.train()
            for input_ids, attention_masks, labels, _ in train_dataloader:
                outputs = self.model(input_ids.to(self.device), attention_masks.to(self.device))

                loss = loss_fn(outputs, labels.to(self.device))
                training_loss += loss.item()

                training_labels += labels.argmax(axis=1).tolist()
                training_preds += outputs.detach().argmax(axis=1).tolist()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()


            validation_loss = 0
            validation_preds = []
            validation_labels = []

            self.model.eval()
            for val_input_ids, val_attention_masks, val_labels, _ in validation_dataloader:
                val_outputs = self.model(val_input_ids.to(self.device), val_attention_masks.to(self.device))

                val_loss = loss_fn(val_outputs, val_labels.to(self.device))
                validation_loss += val_loss.item()

                validation_labels += val_labels.argmax(axis=1).tolist()
                validation_preds += val_outputs.argmax(axis=1).tolist()
            
            training_f1 = f1_score(training_labels, training_preds, average="macro")
            validation_f1 = f1_score(validation_labels, validation_preds, average="macro")
            validation_confusion_matrix = confusion_matrix(validation_labels, validation_preds)
            info = f"Epoch: {epoch}\n" \
                   f"Validation F1 Score:\t{validation_f1}\t|\tValidation Loss: {validation_loss/len(validation_dataloader)}\n" \
                   f"Training F1 Score:\t{training_f1}\t|\tTraining Loss: {training_loss/len(train_dataloader)}\n" \
                   f"Validation Confusion Matrix:\n{validation_confusion_matrix}\n"
            
            logging.info(info)
            print(info)
