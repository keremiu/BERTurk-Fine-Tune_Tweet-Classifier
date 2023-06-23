from src.utils.globals import Globals

from src.services.config_service import ConfigService
from src.services.data_service import DataService

from .model.bert_model import BERTModel
from .data.preprocessor import Preprocessor

import logging

from sklearn.metrics import f1_score, confusion_matrix
import transformers
import platform
import pandas
import shutil
import os

import torch
torch.cuda.empty_cache()        

class Trainer():
    def __init__(self, config_service: ConfigService, data_service: DataService):
        self.config_service = config_service
        self.data_service = data_service
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config_service.bert_model_name)
        
        self.preprocessor = Preprocessor(self.tokenizer)        

        self.device = torch.device(self.config_service.device)

        bert = transformers.AutoModel.from_pretrained(self.config_service.bert_model_name)
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

        # Load model
        MODEL_PATH = self.config_service.model_path if self.config_service.model_path != None else Globals.artifacts_path.joinpath("model", "bert_model.pt")
        self.model.load_state_dict(torch.load(MODEL_PATH))

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
        DEMO_PATH = Globals.artifacts_path.joinpath("demo")
        os.makedirs(DEMO_PATH)
    
        labeled_tweets = self.data_service.read_training_tweets()

        train_dataloader, validation_dataloader, class_weights = self.preprocessor.prepare_inputs(
            labeled_tweets=labeled_tweets, 
            validation_size=self.config_service.validation_size, 
            batch_size=self.config_service.batch_size
        )

        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights)).to(self.device)

        if self.config_service.optimizer == "ADAM":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config_service.learning_rate)
        elif self.config_service.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config_service.learning_rate)
        else:
            raise Exception("The optimizer must be specified either as 'ADAM' or as 'SGD'. See 'fine_tune_tweet_classifier/src/configs/config.yaml'")

        if self.config_service.lr_scheduler != None:
            ... #TODO

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
            validation_texts = []

            self.model.eval()
            for val_input_ids, val_attention_masks, val_labels, val_texts in validation_dataloader:
                val_outputs = self.model(val_input_ids.to(self.device), val_attention_masks.to(self.device))

                try:
                    val_loss = loss_fn(val_outputs, val_labels.to(self.device))
                    validation_loss += val_loss.item()
                except:
                    pass

                validation_labels += val_labels.argmax(axis=1).tolist()
                validation_preds += val_outputs.argmax(axis=1).tolist()
                
                validation_texts += list(val_texts)

            training_f1 = f1_score(training_labels, training_preds, average="macro")
            validation_f1 = f1_score(validation_labels, validation_preds, average="macro")
            validation_confusion_matrix = confusion_matrix(validation_labels, validation_preds)
            info = f"Epoch: {epoch}\n" \
                   f"Validation F1 Score:\t{validation_f1}\t|\tValidation Loss: {validation_loss/len(validation_dataloader)}\n" \
                   f"Training F1 Score:\t{training_f1}\t|\tTraining Loss: {training_loss/len(train_dataloader)}\n" \
                   f"Validation Confusion Matrix:\n{validation_confusion_matrix}\n"
            
            logging.info(info)
            print(info)

            validation_demo = pandas.DataFrame(list(zip( validation_texts, validation_labels, validation_preds)), columns=["text", "label", "prediction"])
            validation_demo.prediction = validation_demo.prediction.apply(lambda x: self.preprocessor.reverse_encoded_label([x])[0])
            validation_demo.label = validation_demo.label.apply(lambda x: self.preprocessor.reverse_encoded_label([x])[0])
            if os.name == "nt":
                validation_demo.to_excel(DEMO_PATH.joinpath(f"epoch{epoch}_demo.xlsx"), index=False)
            elif sys.platform == "darwin":
                validation_demo.to_excel(DEMO_PATH.joinpath(f"epoch{epoch}_demo.xlsx"), index=False)
            elif sys.platform.startswith("linux"):
                validation_demo.to_csv(DEMO_PATH.joinpath(f"epoch{epoch}_demo.csv"), index=False)
        
        # Create the directory for saving the model & the configurations
        model_save_directory = Globals.artifacts_path.joinpath("model")
        os.makedirs(model_save_directory, exist_ok=True)

        # Save the configurations for reproducibility
        shutil.copy(Globals.project_path.joinpath("src", "configs", "config.yaml"), model_save_directory.joinpath("config.yaml"))

        # Save the model's .py file
        shutil.copy(Globals.project_path.joinpath("fine_tune", "model", "bert_model.py"), model_save_directory.joinpath("bert_model.py"))

        # Save the model's .pt file
        torch.save(self.model.state_dict(), model_save_directory.joinpath("bert_model.pt"))