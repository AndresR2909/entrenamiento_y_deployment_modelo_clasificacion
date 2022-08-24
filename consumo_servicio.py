from secrets import randbelow
import requests 
import pandas as pd
import json
import os



Class=[]

data=pd.read_csv(os.path.join(os.getcwd(),'to_predict_2.csv'))
data.drop('Class',axis=1, inplace=True)
print(len(data))
del data[f'Unnamed: 0']
for n in range(len(data)):
    dict=data.iloc[n].to_dict()
    print(dict)
    response = requests.post('http://127.0.0.1:8000/predict', json=dict)
    print()
    jsonToPython = json.loads(response.content)
    Class.append(jsonToPython['prediction'])
    
print(Class)
data['Class']=Class

data.to_csv('to_predict_final.csv')
