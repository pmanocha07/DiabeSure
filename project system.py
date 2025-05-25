# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:40:08 2024

@author: 91708
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open(r"C:\Users\91708\Desktop\mini project sem4\trained_model.sav", 'rb'))
input_data = (4,110,92,0,0,37.6,0.191,30)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
  print('The person is not diabetic')
elif(prediction[0]==1):
  print('The person is diabetic')
else:
  print('error')