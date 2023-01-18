An Electro Cardio Gram (ECG) is simple test that can be used to check your heart rhythm and electrical activity.
Classes = [N:0, S:1, V:2, F:3, Q:4] N: Non-ectopic beats (Normal Beats), - S: Supraventricular ectopic beats , 
V - Ventricular ectopic beats, F - Fusion Beats , Q - Unknown Beats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

train_data = pd.read_csv("path")
test_data = pd.read_csv("path")

train_data

train_data.isnull().sum()

train_data.iloc[:, 187].unique()

train_data[187] = train_data[187].astype('int')
test_data[187] = test_data[187].astype('int')

# Display counts of each classes - Most of Data samples are of normal HeartBeats & its a biased data
sns.catplot(x = 187, kind = 'count', data = train_data)

train_data[187].value_counts()

**Pie chart of each classes**

plt.figure(figsize= (10,10))
my_circle = plt.Circle((0,0), 0.7, color = 'white') 
plt.pie(train_data[187].value_counts(), labels=['Normal Beats','Unknown Beats','Ventricular ectopic beats','Supraventricular ectopic beats',
                                                'Fusion Beats'], autopct = '%0.0f%%', colors = ['red','orange','blue','magenta','cyan'])
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

sns.set_style('whitegrid')
plt.figure(figsize = (20,8))
plt.plot(train_data.iloc[0, 0:187], color = 'red')
plt.show()

# Splitting data into Each Classes
df_1 = train_data[train_data[187] == 1]
df_2 = train_data[train_data[187] == 2]
df_3 = train_data[train_data[187] == 3]
df_4 = train_data[train_data[187] == 4]

sns.set_style("whitegrid")
plt.figure(figsize=(20,8))
plt.plot(train_data.iloc[0, 0:187], color = 'red', label = 'Normal Heart Beats')
plt.plot(df_1.iloc[0, 0:187], color = 'blue', label = 'Supraventricular ectopic beats')
plt.title("ECG Normal vs Supraventricular Ectopic Beats", fontsize = 12)
plt.xlabel("Time (in ms)")
plt.ylabel("Heart Beat Amplitude")
plt.legend()
plt.show()

sns.set_style("whitegrid")
plt.figure(figsize=(20,8))
plt.plot(train_data.iloc[0, 0:187], color = 'red', label = 'Normal Heart Beats')
plt.plot(df_2.iloc[0, 0:187], color = 'blue', label = 'Ventricular ectopic beats')
plt.title("ECG Normal vs Ventricular Ectopic Beats", fontsize = 12)
plt.xlabel("Time (in ms)")
plt.ylabel("Heart Beat Amplitude")
plt.legend()
plt.show()

df_1_upsample = resample(df_1, n_samples = 20000, replace = True, random_state = 123)
df_2_upsample = resample(df_2, n_samples = 20000, replace = True, random_state = 123)
df_3_upsample = resample(df_3, n_samples = 20000, replace = True, random_state = 123)
df_4_upsample = resample(df_4, n_samples = 20000, replace = True, random_state = 123)

# select random 20000 samples from class 0 samples 
df_0 = train_data[train_data[187] == 0].sample(n = 20000, random_state = 123)

#merge all dataframes to create new train samples
train_df = pd.concat([df_0, df_1_upsample, df_2_upsample, df_3_upsample, df_4_upsample])

plt.figure(figsize= (10,10))
my_circle = plt.Circle((0,0), 0.7, color = 'white') 
plt.pie(train_df[187].value_counts(), labels=['Normal Beats','Unknown Beats','Ventricular ectopic beats','Supraventricular ectopic beats',
                                                'Fusion Beats'], autopct = '%0.0f%%', colors = ['red','orange','blue','magenta','cyan'])
p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

# target Y
target_train = train_df[187]
target_test = test_data[187]

target_train.unique()

# convert_integer values into categorical one hot encoding
# 1 - [0, 1, 0, 0, 0]
# 4 - [0, 0, 0, 0, 1]

from keras.utils.np_utils import to_categorical
y_train = to_categorical(target_train)

y_test = to_categorical(target_test)

X_train = train_df.iloc[:, :-1].values
X_test = test_data.iloc[:, :-1].values

# For Conv1D dimentionality must be 187x1 where 187 = number of features , 1 = 1D Dimentionality of Data
X_train = X_train.reshape( len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape( len(X_test), X_test.shape[1], 1)

from keras.models import Sequential
# For F.C. layer
from keras.layers import Dense
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam
# Avoid overfitting by normalizing samples 
from keras.layers.normalization import BatchNormalization

def build_model():
    model = Sequential()
    # Filters = Units in Dense Total number of Neurons
    # Padding = 'same' , zero-padding, Add zero pixels all around input data
    model.add(Conv1D(filters = 64, kernel_size = 6, activation='relu', padding = 'same', input_shape = (187, 1)))
    # Normalization to avoid overfitting
    model.add(BatchNormalization())
    # Pooling 
    model.add(MaxPooling1D(pool_size=(3), strides = (2), padding = 'same'))

    model.add(Conv1D(filters = 64, kernel_size = 6, activation='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides = (2), padding = 'same'))

    model.add(Conv1D( filters = 64, kernel_size = 6, activation='relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides = (2), padding = 'same'))

    # Flatten 
    model.add(Flatten())

    # FC Layer
    # input layer
    model.add(Dense(units = 64, activation='relu'))
    # Hidden Layer
    model.add(Dense(units = 64, activation='relu'))
    # Output Layer
    model.add(Dense(units = 5, activation='softmax'))

    # loss = 'categorical_crossentropy'
    model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

model = build_model()

model.summary()

history = model.fit(X_train, y_train, epochs = 15, batch_size = 32, validation_data=(X_test, y_test))

# evaluate ECG Test Data

model.evaluate(X_test, y_test)

pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()

**Classification Report**

# Make prediction
predict = model.predict(X_test)

# Predicted o/p will be in probability distribution 
predict

# distributional probability to integers
yhat = np.argmax(predict, axis = 1)

from sklearn.metrics import classification_report, confusion_matrix
confusion_matrix(np.argmax(y_test, axis = 1), yhat)

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(np.argmax(y_test, axis = 1), yhat), annot = True, fmt = '0.0f', cmap ='RdPu')

print(classification_report(np.argmax(y_test, axis = 1), yhat))

