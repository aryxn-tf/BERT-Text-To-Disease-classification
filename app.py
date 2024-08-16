from flask import Flask,render_template,request
import pickle
import numpy as np
import traceback
from inference import Inference
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc

sym = pd.read_csv('./data/raw_data/symptom_severity.csv')
sym_list = list(sym['Symptom'].values)
print("Total Known Symptom:- ",len(sym_list))


def pre_process(final_features):
    for i in final_features:
        i = i.replace('_',' ').replace('  ',' ').lower()
    return " ".join(final_features)

def model_results(sample_text):
    res = []
    try:
        model_path = "bert-base-uncased"
        local_model_dict = "./model/model_bert-base-uncased_fold-1"
        max_length = 64
        num_classes = 41
        inf = Inference(model_path,local_model_dict,num_classes,max_length)

        encoder = LabelEncoder()
        encoder.classes_ = np.load('label_enc_classes.npy',allow_pickle=True)

        out = inf.get_results(sample_text)
        res = []
        for i in range(len(out)):
            if i==6:
                break
            dis_name = encoder.inverse_transform([out[i][1]])
            res.append([dis_name[0],out[i][0]])
        del inf
        del encoder
    except Exception as e:
        print("Error:-",str(e))
        traceback.print_exc()
    gc.collect()
    return res

def get_dis_info(dis_result):
    res = []
    desc = pd.read_csv("./data/unique_diseases_info.csv")
    for i in range(len(dis_result)):
        # print(dis_result[i])
        temp = desc[desc['dis_name']==str(dis_result[i][0])]
        dis_desc = temp['dis_description'].values[0]
        dis_symp = temp['dis_symptoms'].values[0]
        dis_prec = temp['dis_precaution'].values[0]
        res.append([dis_result[i][0],dis_result[i][1],dis_desc,dis_symp,dis_prec])
    
    return res


app = Flask(__name__,static_folder='templates/static',)

@app.route('/')
def index():
    return render_template("index.html",sym_list=sym_list)


@app.route('/predict',methods=['POST'])
def predict():
    dis_res,dis_info = [],[]
    try:
        features = [x for x in request.form.values()]
        features = pre_process(features)
        dis_res = model_results(features)

        dis_info = get_dis_info(dis_res)
        
    except Exception as e:
        print('Error predict()->',str(e))
        traceback.print_exc()

    return render_template("result.html",res_dis_name=dis_info,res_dis_desc=dis_info)


if __name__ == '__main__':
    app.run(debug=True)