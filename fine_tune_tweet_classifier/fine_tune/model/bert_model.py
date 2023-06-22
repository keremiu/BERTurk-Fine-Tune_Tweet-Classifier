import transformers
import torch.nn

class BERTModel(torch.nn.Module):
    def __init__(self, bert: transformers.BertModel, layer_infos: list[dict]):
        super(BERTModel, self).__init__()

        self.bert = bert
        
        self.layers = [
            self.get_layer(layer_info)
            for layer_info in layer_infos
        ]

    def get_layer(self, layer_info: dict):            
        if layer_info["name"] == "linear":
            layer = torch.nn.Linear(layer_info["in"], layer_info["out"])
    
        elif layer_info["name"] == "dropout":
            layer = torch.nn.Dropout(layer_info["p"])

        elif layer_info["name"] == "softmax":
            layer = torch.nn.Softmax()
        elif layer_info["name"] == "relu":
            layer = torch.nn.ReLU()

        return layer

    def forward(self, input_ids, attention_mask):
        _, output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)

        for layer in self.layers:
            output = layer(output)

        return output