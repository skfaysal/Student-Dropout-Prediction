import pickle
import pandas as pd
import numpy as np
#load encoder
enc = './saved models/encoder.pkl'
encoder = pickle.load(open(enc, 'rb'))

#load scaler
scl = './saved models//sc.pkl'
scaler = pickle.load(open(scl, 'rb'))

#load model
mod = './saved models//model_logreg.pkl'
model = pickle.load(open(mod, 'rb'))

mod1 = './saved models//model_randomforest.pkl'
randomforest = pickle.load(open(mod, 'rb'))


test_df=pd.DataFrame({'ATTENDANCE':[7],
                     'GRAND_TOTAL':[70],
                     'GRADE_POINT':[3.2],
                     'FK_CAMPUS':['C01'],
                     'ATTENDANCEPERCENTAGE':[0],
                     'SGPA':[1.77],
                      'CREDIT':[4200],
                      'PAYABLE':[458050],
                      'PAID':[419075],
                      'DUE':[38975],
                      'CGPA':[2.21],
                      'SSC_CGPA': [3.5],
                      'HSC_CGPA': [3],
                      'RELIGION': ['Islam'],
                      'BEAREDUEXPENSE': ['Not given'],
                      'ENROLLMENT_AGE': [17],
                      'FATHERANNUALINCOME': [0],
                      'MOTHERANNUALINCOME': [0],
                      'MARITAL_STATUS': ['Single'],
                      'SEX': ['FeMale']
                      })


print(type((encoder)))
print(encoder)
test_enc=test_df.reindex(columns=encoder,fill_value=0)
print(test_enc)
#test_enc.to_csv("C:/Users/nipu/OneDrive/Desktop/dropout_new/test.csv")
#print(test_enc.columns)
a=np.array(test_enc)
print(a)
data = scaler.transform(a)
print(data)

#prediction = model.predict(data)
#print(prediction)