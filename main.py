import numpy as np
import pandas as pd

import talib

#epochs_list = [500, 1000, 1500, 2000]
#first_layer_param = [50, 100, 150, 200]
#second_layer_param = [500, 1000, 1500, 2000]
epochs_list = [5, 10]
first_layer_param = [500, 100]
second_layer_param = [50, 100]

# default df
df = pd.read_csv("data.csv")
df.isna().sum()
df.drop(["Unnamed: 0", "symbol"], axis=1, inplace=True)
df.set_index(['open_time'], inplace=True)

df[f'rsi_7'] = talib.RSI(df.close, timeperiod=7)
df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df = df.iloc[33:]


X, y = df.drop(columns=['close']), df.close.values

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normal_data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
#normal_data = df

n=120
feature_names = list(normal_data.drop('close', axis=1).columns)
X = []
y = []

normal_data_X = normal_data[feature_names]
for i in range(0, len(normal_data)-n):
  X.append(normal_data_X.iloc[i:i+n].values)
  y.append(normal_data['close'].iloc[i+n-1])

X = np.array(X)
y = np.array(y)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
n_steps = n
n_features = 9

X_train = X[0:3600]
y_train = y[0:3600]

result = []

print("Start training models")
for e in epochs_list:
  for i in range(len(first_layer_param)):
    epochs = e
    layer1 = first_layer_param[i];
    layer2 = second_layer_param[i]

    print("============ LAYERS PARAMS============")
    print(f"Epochs: {epochs}")
    print(f"Layer 1: Dense with {layer1}")
    print(f"Layer 2: LSTM with {layer2}")
    print()

    model = Sequential()
    model.add(Dense(layer1))
    model.add(LSTM(layer2,activation='relu', return_sequences=False, input_shape = (n_steps, n_features)))
    model.add(Dense(1))
    #model.summary()

    model.compile(optimizer = 'adam', loss = 'mse', metrics=['mse', 'mape'])
    history = model.fit(X_train, y_train, batch_size=180, epochs=epochs, verbose=1)

    model.save(f'models/epochs{epochs}_{n_steps}_{n_features}_D{layer1}_L{layer2}')

    import matplotlib.pyplot as plt
    plt.title('train & loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')

    plt.plot(history.history['loss'])
    print(history.history['loss'])
    start = len(X)-1830
    end = len(X)-30

    X_test = X[start:end]
    predict = model.predict(X_test)
    predictions = pd.DataFrame(predict).rename(columns={0: 'Predict'})

    y_test = y[start:end]
    y_test = pd.DataFrame(y_test).rename(columns={0: 'Real'})


    final = pd.concat([predictions,y_test],axis=1)
    final['mae'] = abs(final['Predict'] - final['Real'])
    print(final)
    print(final['mae'].mean())

    avg_mae_mean = f"{(final['Predict'].mean()-final['Real'].mean()) / final['Real'].mean() * 100}%"

    #norm_data = pd.DataFrame(scaler.inverse_transform(norm_data), columns=df.columns, index=df.index)
    nd = normal_data
    for i in range(len(predict)):
      nd.iloc[start+i+1+n]['close'] = predict[i][0]

    nd = pd.DataFrame(scaler.inverse_transform(nd), columns=df.columns, index=df.index)

    # 預計 - 實際
    a = pd.concat([(nd['close'][start-30:end+30]).rename("Predict"), (df['close'][start-30:end+30]).rename("Real")],axis=1)
    nd['close'][start-30:end+30].plot(color='#00ff00') # 預計
    df['close'][start-30:end+30].plot(color='#FF0000') # 實際

    avg_mae2_mean = f"{(a['Predict'].mean() - a['Real'].mean()) / a['Real'].mean() * 100}%"

    a.to_excel(f"xls/epochs{epochs}_{n_steps}_{n_features}_D{layer1}_L{layer2}.xlsx")
    # a.plot()
    result.append({
      'layer1': layer1,
      'layer2': layer2,
      'epochs': epochs,
      'avg. mae mean': avg_mae_mean,
      'avg. Actual mean': avg_mae2_mean
    })

pd.DataFrame(result).to_excel("xls/result.xlsx")
