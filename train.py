import numpy as np

df_x = np.load("Dados Normalizados.npy")
print(df_x.shape)
df_y = np.load("Target.npy")
print(df_y.shape)
df = np.concatenate([df_x, np.expand_dims(df_y,axis=-1)], axis=-1)
#df = np.expand_dims(df, axis = -1)
#df = df.reshape(df.shape[0], 1, df.shape[1], df.shape[2])
print(df.shape)

import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2" 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, Conv2D, Activation, MaxPooling2D, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, SimpleRNN
from keras.utils import multi_gpu_model
from keras import backend as K
K.set_image_dim_ordering('tf')

kf = KFold(n_splits=5)

np.random.seed(10)
print(df[:5,0,-1])
np.random.shuffle(df)
print(df[:5,0,-1])

split = 1
for percentage in range(1,10):
    if split != 1:
        continue
    else:

        embed_dim = 128
        lstm_out = int(np.floor(486*4//1))
        batch_size = 32
        activation = 'tanh'
        print(split)
        inp = Input(shape=(df.shape[1], 2))
        #model.add(Embedding(2500, embed_dim,input_length = df.shape[1] - 1, dropout = 0.2))
        
        #x = Conv1D(128, 3, padding='same')(inp)
        #x = BatchNormalization()(x)
        #x = Activation(activation)(x)
        #x = MaxPooling1D(2, padding='same')(x)     
        #x = Conv1D(256, 3, padding='same')(x)
        #x = BatchNormalization()(x)
        #x = Activation(activation)(x)
        #x = MaxPooling1D(2, padding='same')(x)
        #x = Conv1D(512, 3, padding='same')(x)
        #x = BatchNormalization()(x)
        #x = Activation(activation)(x)
        #x = MaxPooling1D(2, padding='same')(x)
        #x = Conv2D(32, (2,2), padding='same')(x)
        #x = BatchNormalization()(x)
        #x = Activation(activation)(x)
        #x = MaxPooling2D(2, padding='same')(x)
        #print(x.shape)
        #x = Reshape((851,32))(x)
        #x = Flatten()(x)
        #x = LSTM(lstm_out, return_sequences=True)(x)
        x = LSTM(lstm_out)(inp)
        x = Dense(lstm_out//2, activation=activation)(x)
        #x = Dense(lstm_out, activation=activation)(x)
        #x = Dense(100, activation='relu')(x)

        x = Dense(2,activation='sigmoid')(x)
        #x = Reshape((2,1))(x)

        model = Model(inp, x)
        #parallel_model = multi_gpu_model(model, gpus=5)
        parallel_model = model
        parallel_model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
        #print(parallel_model.summary())

        #print("TRAIN:", train_index, "TEST:", test_index)
        #X_train, X_test = df[train_index,:,:2], df[test_index,:,:2]
        #Y_train, Y_test = df[train_index,0,-1], df[test_index,0,-1]
        X_train, X_test, Y_train, Y_test = train_test_split(df[:,:,:2], df[:,0,-1], test_size = 0.10, random_state = 7)
        
        Y_test_binalized = np.zeros((Y_test.shape[0],2), dtype="float32")

        for i in range(Y_test.shape[0]):
            if Y_test[i] == 0.:
                Y_test_binalized[i,0] = 1.
            else:
                Y_test_binalized[i,1] = 1.
        
        Y_train_binalized = np.zeros((Y_train.shape[0],2), dtype="float32")

        for i in range(Y_train.shape[0]):
            if Y_train[i] == 0.:
                Y_train_binalized[i,0] = 1.
            else:
                Y_train_binalized[i,1] = 1.
                
        
                
        #print(Y_train.shape)
        #tensorboard = TensorBoard(log_dir=f'./fold{split}')

        parallel_model.fit(X_train, Y_train_binalized, batch_size =batch_size*1, epochs = 5,  
                       verbose = 1, validation_data=(X_test, Y_test_binalized))

        pred = parallel_model.predict(X_test)
        
        '''
        label_binalized = np.zeros((Y_test.shape[0],2), dtype="float32")

        for i in range(Y_test.shape[0]):
            if Y_test[i] == 0.:
                label_binalized[i,0] = 1.
            else:
                label_binalized[i,1] = 1.

        pred_binalized = np.zeros((pred.shape[0],2), dtype="float32")

        for i in range(pred.shape[0]):
            if pred[i] < 0.5:
                pred_binalized[i,0] = 1.
            else:
                pred_binalized[i,1] = 1.
        '''
    
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(Y_test_binalized[:, i], pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])


        plt.figure()
        lw = 2
        colors = ['darkblue','darkorange']
        classes = ['AGN','Blazar',]
        for i in range(2):
            plt.plot(fpr[i], tpr[i], color=colors[i],
                     lw=lw, label=f'{classes[i]} (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - Final Model')
        plt.legend(loc="lower right")
        #plt.savefig(f'Train_Test/ROC-NO LSTM.jpg')
        plt.show()

        split += 1