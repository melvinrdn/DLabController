import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def plot_from_csv(csv_file,skip=0):
    df = pd.read_csv(csv_file)
    print(df.shape)
    data_array = df.values[0:, skip:].astype(float)
    print(data_array.shape)
    print('--')
    return data_array


data_original = plot_from_csv('wavefront_correction_infrared_SLM.csv',skip=1)
data_corrected = plot_from_csv('C_interpolated.csv',skip=1)
data_adjusted = (data_original+data_corrected)
plt.figure()
plt.imshow(data_adjusted , cmap='bwr')
plt.colorbar()
plt.title('Plot from CSV')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

df = pd.DataFrame(data_adjusted.astype(int))
df.to_csv('C_test.csv', index=False, header=False)