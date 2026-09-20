
import os
import pandas as pd
import numpy as np



import tensorflow as tf
print("Gpu is available:", tf.config.list_physical_devices('GPU'))

tf.config.optimizer.set_jit(False)



print("library imported")
model_data=pd.read_csv("/home/users/bipink/NN_Model_Hyperlocal/review2/Input_data/Full_Input_10_point_except_9596011314.csv")  
model_data.dropna(inplace=True)
print(model_data['0'].values)
print(model_data.shape)
print("imported dataset")
#--------------------------------------
#Wind speed absolute values
# Considering only postive values of wind data
#--------------------------------------

model_data['8']=model_data['8'].abs()
model_data['9']=model_data['9'].abs()
model_data['18']=model_data['18'].abs()
model_data['17']=model_data['17'].abs()
model_data['27']=model_data['27'].abs()
model_data['26']=model_data['26'].abs()
model_data['36']=model_data['36'].abs()
model_data['35']=model_data['35'].abs()

model_data['45']=model_data['45'].abs()
model_data['44']=model_data['44'].abs()
model_data['54']=model_data['54'].abs()
model_data['53']=model_data['53'].abs()
model_data['63']=model_data['63'].abs()
model_data['62']=model_data['62'].abs()
model_data['72']=model_data['72'].abs()
model_data['71']=model_data['71'].abs()
model_data['81']=model_data['81'].abs()
model_data['80']=model_data['80'].abs()
model_data['90']=model_data['90'].abs()
model_data['89']=model_data['89'].abs()
print("col7",model_data['7'])
print("col8",model_data['8'])
#--------------------------------------

#Define Input and Labels, Train-Validation set split

x=model_data.drop(['94','0'],axis=1).values
y=model_data['94']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#---------------------------------------
#Normalizing the data using MinMaxScaler
#---------------------------------------
from sklearn.preprocessing import MinMaxScaler
scaler_x=MinMaxScaler()
scaler_y=MinMaxScaler()

X_train=scaler_x.fit_transform(X_train)

y_train=y_train.values.reshape(-1,1)
y_train=scaler_y.fit_transform(y_train)

X_test=scaler_x.transform(X_test)

y_test=y_test.values.reshape(-1,1)
y_test=scaler_y.transform(y_test)

print("Preprocessing complete")

#---------------------------------------
del x
del y
del model_data


#---------------------------------------
#Define the Neural Network model
#---------------------------------------

ann=tf.keras.models.Sequential()

#This is the 256 units job
ann.add(tf.keras.layers.Dense(units=256,input_shape=(93,), activation='PReLU'))

ann.add(tf.keras.layers.Dense(units=162, activation='PReLU'))
#ann.add(tf.keras.layers.Dropout(0.2))
ann.add(tf.keras.layers.Dense(units=81, activation='PReLU'))

#ann.add(tf.keras.layers.Dropout(0.2))
ann.add(tf.keras.layers.Dense(units=45, activation='PReLU'))
ann.add(tf.keras.layers.Dense(units=45, activation='PReLU'))
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])

#for early stopping based on validation loss
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
mc = ModelCheckpoint("/home/users/bipink/NN_Model_Hyperlocal/review2/model_training/model_with_9596011314_as_testing/gen_models/model_with_para_10p", monitor='val_loss', mode='min', verbose=1, save_best_only=True)

print("Neural Network defined")


history=ann.fit(X_train, y_train, batch_size=2048, epochs=1000, validation_data=(X_test, y_test),callbacks=[es, mc])

import pickle

# Save history as a pickle file
with open('/home/users/bipink/NN_Model_Hyperlocal/review2/model_training/model_with_9596011314_as_testing/model_training_history/history_model_with_para_10.pkl', 'wb') as f:
    pickle.dump(history.history, f)
