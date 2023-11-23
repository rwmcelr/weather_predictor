import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

def process_data(weatherCSV):
  weatherCSV = 'central_park.csv'
  dat_init = pd.read_csv(weatherCSV, index_col = 'DATE')
  dat_filtered = dat_init[['PRCP','TMAX','TMIN','SNOW','SNWD']]
  dat_filtered.columns = ['precipitation', 'max_temp', 'min_temp', 'snowfall', 'snowdepth']
  dat_filtered.index = pd.to_datetime(dat_filtered.index)
  zero_values = dat_filtered.apply(pd.isnull).sum()
  
  if zero_values['max_temp'] > 0 or zero_values['min_temp'] > 0 = dat_filtered[['max_temp','min_temp']].ffill(inplace = True)
  zero_percentage = zero_values / dat_filtered.apply(len)
  for col in dat_filtered.columns:
    if zero_percentage[col] > 0.15: dat_filtered.drop(col)
  # insert for loop to remove columns with > 10% total NAs
  # dat_filtered[['precipitation', 'snowfall', 'snowdepth']] = dat_filtered[['precipitation', 'snowfall', 'snowdepth']].fillna(value = 0)