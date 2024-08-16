# Text-to-Disease-Classification

Formulated a large Language Model (LLM) by fine tuning pre-trained BERT transformer ,that can accurately classify disease name based on short description of symptoms.

Deployed this system as a Flask-based web application, ensuring seamless accessibility and user-friendly interaction.

![index](/templates/static/flash_index.png)

## Installation

Create a conda environment using environment file , activate that environment and the run the app.py file.
if run sucessfully, you should be able to access the web app at localhost.

Example:-   
```
# clone repo
git clone https://github.com/aryxn-tf/BERT-Text-To-Disease-classification

# create environment
conda env create --name <envname> --file=environments.yml

# activate env
conda activate <envname>

# run app.py
python app.py

```

## Data

Used below kaggle dataset to fine tune the BERT model

Kaggle:// [disease-symptom-description-dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)

#### notes :-
    -  Data is equally Balanced, 120 symptom lists/entries for each 43 disease.
    -  all disease have corresponding description and precautions
    -  visualised the count plot of individual symptom.
        -  type of symptoms are more balanced and diverse in symptom1 and symptom2 feature column.
    -  removed all chars other then aplha-numeric. removed extra spaces. 

## Training & Evaluation
    -  Used OOP by defining custom Dataset and Model class to load dataset and fine tune BERT model.
    -  Used K fold cross Validation for Training and evaluation of model.
    -  Trained the model for 3 epochs for 5 folds

#### notes :-
    - FOLD 1 , Epoch: 2 ,loss : 3.163252592086792 , accuracy :0.991869918699187
    - FOLD 2 , Epoch: 2 ,loss : 3.164623737335205 , accuracy :0.9956808943089431
    - FOLD 3 , Epoch: 2 ,loss : 3.1687734127044678 , accuracy :0.9852642276422764
    - FOLD 4 , Epoch: 2 ,loss : 3.1720407009124756 , accuracy :0.9933943089430894
    - FOLD 5 , Epoch: 2 ,loss : 3.1956565380096436 , accuracy :0.9784044715447154

## Final Results

when you enter the top 7 symptoms in the flask web app , it will display the top 5 disease based on the symptoms.

Here is the sample result from flask , 
![result](BERT-Text-to-Disease-Classification/templates/static/flask_result.png "result html")
