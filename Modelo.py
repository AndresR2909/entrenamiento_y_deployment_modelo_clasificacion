import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from joblib import load
from pydantic import BaseModel

import os



# clase que describe las entradas
class ClasificadorComprasDatos(BaseModel):
    autoID: str
    SeniorCity: str
    Partner: str
    Dependents: str
    Service1: str
    Service2: str
    Security: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    Contract: str
    PaperlessBilling : str
    PaymentMethod: str
    Charges: float
    Demand: float


#lase que carga procesamiento y hace predicciones
class ClasificadorCompras:

    def __init__(self):
        self.model_fname_ = os.path.join(os.getcwd(),'modelo_clasificador.joblib')
        self.procesamiento_fname_= os.path.join(os.getcwd(),'preprocessor.joblib')
        self.model = load(self.model_fname_)
        self.procesamiento = load(self.procesamiento_fname_)


    # Predecir
    # 
    def predict_compras(self, autoID, SeniorCity, Partner, Dependents, Service1, Service2, Security, OnlineBackup, DeviceProtection , TechSupport , Contract , PaperlessBilling , PaymentMethod , Charges, Demand):
        data_in = [[autoID, SeniorCity, Partner, Dependents, Service1, Service2, Security, OnlineBackup, DeviceProtection , TechSupport , Contract , PaperlessBilling , PaymentMethod , Charges, Demand]]
        
        columns= ['autoID','SeniorCity','Partner','Dependents','Service1','Service2','Security','OnlineBackup','DeviceProtection','TechSupport','Contract','PaperlessBilling','PaymentMethod','Charges','Demand']
        numeric_cols=['Charges', 'Demand']
        cat_cols=['SeniorCity','Partner','Dependents','Service1','Service2', 'Security', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'Contract','PaperlessBilling', 'PaymentMethod']
        
        x_in=pd.DataFrame(data_in,columns=columns)
        x_in.drop('autoID', axis=1, inplace=True)

        #aplicar transformaciones
        encoded_cat = self.procesamiento.named_transformers_['cat']['onehot']\
              .get_feature_names(cat_cols)
        labels = np.concatenate([numeric_cols, encoded_cat])
        X_in_p = self.procesamiento.transform(x_in)
        X_in_p = pd.DataFrame(X_in_p, columns=labels)

        #eliminar columnas dependientes
        X_in_p.drop(['TechSupport_No internet service','DeviceProtection_No internet service', 'OnlineBackup_No internet service'], axis = 'columns', inplace=True)
        
        prediction = self.model.predict(X_in_p)

        if prediction==0:
            Class='Alpha'
        else:
            Class='Beta'
  
        return Class