import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import transformers
import pandas as pd
import numpy as np
import time
import gc
import traceback
import datetime
from tqdm import tqdm

from dataset import TestDataset
from model import Model



'''
 self.model defined here is final output which includes pre trained bert and fc layer we added
 self.model defined in Model class is only output of pretrained model 
'''

class Inference():
    def __init__(self, model_path,local_model_dict, num_classes,max_length):
        self.local_model_dict = local_model_dict
        self.max_length = max_length
        self.model_path = model_path
        self.num_classes = num_classes
        self.model = Model(model_path=self.model_path,num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(self.local_model_dict))
        
    def get_results(self,raw_text,test_batch_size=1):
        test_data = TestDataset(raw_text,self.model_path,self.max_length)
        test_data_loader = torch.utils.data.DataLoader(test_data,batch_size=test_batch_size,shuffle=True)

        return self._evaluate_model(test_data_loader)


    def _evaluate_model(self,test_data_loader):
        try:
            self.model.eval()
            for step,batch in enumerate(test_data_loader):
                batch_inputs, batch_masks = batch['ids'], batch['mask']
                batch_token_type_ids = batch['token_type_ids']
                with torch.no_grad():
                    outputs = self.model(batch_inputs, batch_masks, batch_token_type_ids)
                # out = [[np.argmax(i),i] for i in outputs.cpu().detach().numpy()]
                out = list(outputs.cpu().detach().numpy()[0])
                out = [[out[i],i] for i in range(len(out))]
            return sorted(out,reverse=True)
        except Exception as e:
            print("Error in evaluate_model >>>>>>")
            traceback.print_exc()
            return None



if __name__ == "__main__":
    sample_text = "continuous sneezing  shivering  chills  watering from eyes"
    sample_text_1 = "itching  yellowish skin  nausea  loss of appetite  abdominal pain  yellowing of eyes"

    model_path = "bert-base-uncased"
    local_model_dict = "./model/model_bert-base-uncased_fold-1"
    max_length = 64
    dis = pd.read_csv("./data/unique_diseases_info.csv")
    num_classes = len(list(dis['dis_name'].values))
    del dis
    gc.collect()

    inf = Inference(model_path,local_model_dict,num_classes,max_length)

    res1 = inf.get_results(sample_text)
    # res2 = inf.get_results(sample_text_1)
    for i in res1:
        print(i)

    del inf
    gc.collect()
