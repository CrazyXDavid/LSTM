import pandas as pd
import matplotlib.pyplot as plt
from os import path
import sklearn.metrics as met
import numpy as np

epochs_list = [500, 1000, 1500, 2000]
first_layer_param = [50, 100, 150, 200, 250]
second_layer_param = [25, 50, 75, 100, 150]

result = []
for e in epochs_list:
    for flp in first_layer_param:
        for slp in second_layer_param:
            epochs = e
            layer1 = flp
            layer2 = slp

            xlsxFile = f'xls/epochs{epochs}_120_9_D{layer1}_L{layer2}.xlsx'
            # print(xlsxFile)
            if (path.isfile(xlsxFile)):
                df = pd.read_excel(xlsxFile)
                predictions, real = df['Predict'], df['Real']
                avg_mae_mean = f"{(predictions.mean() - real.mean()) / real.mean() * 100}%"
                # print(f'{epochs}_{layer1}_{layer2}_mae_{avg_mae_mean}')

                np_predict, np_real = np.array(predictions), np.array(real)
                
                mean_absolute_error = met.mean_absolute_error(
                    np_real, np_predict)
                #print(f'{epochs}_{layer1}_{layer2}_mse{mean_absolute_error}')
                
                mean_absolute_percentage_error = met.mean_absolute_percentage_error(
                    np_real, np_predict)
                # print(f'{epochs}_{layer1}_{layer2}_maep_{mean_absolute_percentage_error}')
                
                mean_squared_error = met.mean_squared_error(
                    np_real, np_predict)
                # print(f'{epochs}_{layer1}_{layer2}_mse_{mean_squared_error}')
                
                
                result.append({
                    'layer1': layer1,
                    'layer2': layer2,
                    'epochs': epochs,
                    'avg. mae mean': avg_mae_mean,
                    'Mean abs error': mean_absolute_error,
                    'Mean abs error %': mean_absolute_percentage_error,
                    'Squared error': mean_squared_error
                })
                
#print(result)
pd.DataFrame(result).to_excel("xls/result.xlsx")
                
                
                
