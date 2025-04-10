from statistics import correlation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_FAWA = pd.read_csv(f'./QFINdata/FAWA.csv')
df_SMIF = pd.read_csv(f'./QFINdata/SMIF.csv')
df_FAWA['Mid'] = (df_FAWA['Bids'] + df_FAWA['Asks']) / 2
df_SMIF['Mid'] = (df_SMIF['Bids'] + df_SMIF['Asks']) / 2
midFAWA = np.array(df_FAWA['Mid'])
midSMIF = np.array(df_SMIF['Mid'])
midSMIF = midSMIF.reshape(-1,1)
midFAWA = midFAWA.reshape(-1,1)
scaledFAWA = scaler.fit_transform(midFAWA)
scaledSMIF = scaler.fit_transform(midSMIF)



spread = scaledFAWA - scaledSMIF
div = midSMIF/midFAWA

# results = seasonal_decompose(spread, period=7000)
# plt.plot(results.trend)
# # plt.plot(df['Mean100'])
#plt.show()
plt.plot( midSMIF, color='red')
#plt.plot(df['Mid'], color='blue')
plt.show()

# Find the index where the maximum occurs
# max_index = results.seasonal.idxmax()
# min_index = results.seasonal.idxmin()
# print(f"at index {max_index}.")
# print(f"at index {min_index}.")
# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(df['Mid'], lags=280)
# plt.show()

#
# plt.plot(results.trend)
# plt.show()
# plt.plot(df['Std'])
# plt.show()