import torch
import torch.nn as nn
import transformers

'''
 self.model defined here is exclusive to Model class which is only output of bert
 model that we define and use in Inference class is the final output which includes the fc layers as well
'''

class Model(nn.Module):
    def __init__(self,model_path,num_classes):
        super().__init__()
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = transformers.AutoModel.from_pretrained(self.model_path)
        self.fc_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(768,self.num_classes),
            nn.Sigmoid()
        )
    
    def forward(self,ids,mask,token_type_ids,targets=None):
        out = self.model(input_ids=ids,attention_mask=mask,token_type_ids=token_type_ids)
        # out a dict with 2 keys ..odict_keys(['last_hidden_state', 'pooler_output'])
        pooler_output = out.get('pooler_output')
        # pooler output dimension is batch_size*768 (thi model ops 1x768 encoding)
        fc_out = self.fc_layer(pooler_output)
        # fc out dimension is batch_size*(num_classes)
        return fc_out