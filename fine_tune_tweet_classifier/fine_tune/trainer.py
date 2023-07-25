from src.utils.globals import Globals

from src.services.config_service import ConfigService
from src.services.data_service import DataService

from .model.bert_model import BERTModel
from .data.preprocessor import Preprocessor
import torch.optim.lr_scheduler as lr_scheduler

import logging

from sklearn.metrics import f1_score, multilabel_confusion_matrix
import transformers
import pandas
import shutil
import sys
import os

from tqdm import tqdm

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

    def train(self):
        # Create the directory for saving the model & the configurations
        MODEL_DIR = Globals.artifacts_path.joinpath("model")
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Create the directory for saving the validation demos
        DEMO_DIR = Globals.artifacts_path.joinpath("validation_demo")
        os.makedirs(DEMO_DIR, exist_ok=True)
    
        labeled_tweets = self.data_service.read_training_tweets()

        train_dataloader, validation_dataloader, class_weights = self.preprocessor.prepare_inputs(
            labeled_tweets=labeled_tweets, 
            validation_size=self.config_service.validation_size, 
            batch_size=self.config_service.batch_size
        )

        if self.config_service.class_weights == True:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weights)).to(self.device)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss().to(self.device)

        if self.config_service.optimizer == "ADAM":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config_service.learning_rate)
        elif self.config_service.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config_service.learning_rate)
        else:
            raise Exception("The optimizer must be specified either as 'ADAM' or as 'SGD'. See 'fine_tune_tweet_classifier/src/configs/config.yaml'")

        if self.config_service.lr_scheduler == "LinearLR":
           scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)
        elif self.config_service.lr_scheduler == "ExponentialLR":
           scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
           
        best_validation_f1 = 0
        epochs_no_improve = 0
        for epoch in range(1, self.config_service.num_epochs+1):
            training_loss = 0
            training_preds = []
            training_labels = []
            patience = self.config_service.patience  # Make sure to set this value in your config
            
            self.model.train()
            for input_ids, attention_masks, labels, _ in tqdm(train_dataloader, desc=f'Training Epoch {epoch}', unit='batch'):
                outputs = self.model(input_ids.to(self.device), attention_masks.to(self.device))

                loss = loss_fn(outputs, labels.float().to(self.device))
                training_loss += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()
                if self.config_service.lr_scheduler != None:
                    scheduler.step()
                training_labels += labels.tolist()
                outputs[outputs >= self.config_service.threshold] = 1
                outputs[outputs < self.config_service.threshold] = 0
                training_preds += outputs.int().tolist()


            validation_loss = 0
            validation_preds = []
            validation_labels = []
            validation_texts = []

            self.model.eval()
            for val_input_ids, val_attention_masks, val_labels, val_texts in tqdm(validation_dataloader, desc=f'Validation Epoch {epoch}', unit='batch'):
                val_outputs = self.model(val_input_ids.to(self.device), val_attention_masks.to(self.device))

                val_loss = loss_fn(val_outputs, val_labels.float().to(self.device))
                validation_loss += val_loss.item()

                validation_labels += val_labels.tolist()
                val_outputs[val_outputs >= self.config_service.threshold] = 1
                val_outputs[val_outputs < self.config_service.threshold] = 0
                validation_preds += val_outputs.int().tolist()
                
                validation_texts += list(val_texts)

            training_f1 = f1_score(training_labels, training_preds, average="macro")
            validation_f1 = f1_score(validation_labels, validation_preds, average="macro")
            validation_confusion_matrix = multilabel_confusion_matrix(validation_labels, validation_preds)
            info = f"Epoch: {epoch}\n" \
                   f"Validation F1 Score:\t{validation_f1}\t|\tValidation Loss: {validation_loss/len(validation_dataloader)}\n" \
                   f"Training F1 Score:\t{training_f1}\t|\tTraining Loss: {training_loss/len(train_dataloader)}\n" \
                   f"Validation Confusion Matrix:\n{validation_confusion_matrix}\n"
            
            logging.info(info)
            print(info)

            validation_demo = pandas.DataFrame(list(zip( validation_texts, validation_labels, validation_preds)), columns=["text", "label", "prediction"])
            validation_demo.prediction = validation_demo.prediction.apply(lambda x: self.preprocessor.decode_labels(x))
            validation_demo.label = validation_demo.label.apply(lambda x: self.preprocessor.decode_labels(x))
            if os.name == "nt" or sys.platform == "darwin":
                validation_demo.to_excel(DEMO_DIR.joinpath(f"epoch{epoch}_demo.xlsx"), index=False)
            else:
                validation_demo.to_csv(DEMO_DIR.joinpath(f"epoch{epoch}_demo.csv"), index=False)
                
            if validation_f1 > best_validation_f1:
                best_validation_f1 = validation_f1
                epochs_no_improve = 0
                # Save the best model
                torch.save(self.model.state_dict(), MODEL_DIR.joinpath("best_model.pt"))
                print("Validation F1 Score improved, saving model...")
            elif patience is not None:  # If patience is None, we never increment epochs_no_improve or stop early
                epochs_no_improve += 1
                print(f"No improvement in Validation F1 Score for {epochs_no_improve} epochs.")

                # If the validation score hasn't improved for 'patience' number of epochs, stop the training
                if epochs_no_improve == patience:
                    print('Early stopping!')
                    break
        # Save the configurations for reproducibility
        shutil.copy(Globals.project_path.joinpath("src", "configs", "config.yaml"), MODEL_DIR.joinpath("config.yaml"))

        # Save the model's .py file
        shutil.copy(Globals.project_path.joinpath("fine_tune", "model", "bert_model.py"), MODEL_DIR.joinpath("bert_model.py"))

        # Save the model's .pt file
        torch.save(self.model.state_dict(), MODEL_DIR.joinpath("bert_model.pt"))

        logging.info(f"Model saved at: '{MODEL_DIR.joinpath('bert_model.pt')}'")
        print(f"Model saved at: '{MODEL_DIR.joinpath('bert_model.pt')}'")

    def test(self):
        DEMO_PATH = Globals.artifacts_path.joinpath("test_demo")
        os.makedirs(DEMO_PATH, exist_ok=True)

        labeled_tweets = self.data_service.read_test_tweets()
        
        test_dataloader, _, _ = self.preprocessor.prepare_inputs(
            labeled_tweets=labeled_tweets, 
            validation_size=1, 
            batch_size=self.config_service.batch_size
        )

        labels = []
        preds = []
        texts = []

        # Load model
        MODEL_PATH = self.config_service.model_path if self.config_service.model_path != None else Globals.artifacts_path.joinpath("model", "bert_model.pt")
        self.model.load_state_dict(torch.load(MODEL_PATH))

        self.model.eval()
        for test_input_ids, test_attention_masks, test_labels, test_texts in tqdm(test_dataloader, desc='Testing', unit='batch'):
            test_outputs = self.model(test_input_ids.to(self.device), test_attention_masks.to(self.device))

            labels += test_labels.tolist()
            test_outputs[test_outputs >= self.config_service.threshold] = 1
            test_outputs[test_outputs < self.config_service.threshold] = 0
            preds += test_outputs.int().tolist()

            texts += list(test_texts)

        test_f1 = f1_score(labels, preds, average="macro")
        test_confusion_matrix = multilabel_confusion_matrix(labels, preds)
        info =  f"Test F1 Score:\t{test_f1}\n" \
                f"Test Confusion Matrix:\n{test_confusion_matrix}\n"
        
        logging.info(info)
        print(info)

        test_demo = pandas.DataFrame(list(zip(texts, labels, preds)), columns=["text", "label", "prediction"])
        test_demo.prediction = test_demo.prediction.apply(lambda x: self.preprocessor.decode_labels(x))
        test_demo.label = test_demo.label.apply(lambda x: self.preprocessor.decode_labels(x))
        if os.name == "nt" or sys.platform == "darwin":
            test_demo.to_excel(DEMO_PATH.joinpath(f"demo.xlsx"), index=False)
        else:
            test_demo.to_csv(DEMO_PATH.joinpath(f"demo.csv"), index=False)

        logging.info(f"Test Demo saved at: '{DEMO_PATH.joinpath('demo.xlsx')}'")
        print(f"Test Demo saved at: '{DEMO_PATH.joinpath('demo.xlsx')}'")