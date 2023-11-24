import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
# Implement more ML algorithms

def process_data(weatherCSV):
  ''' Might be a valid consideration to only run this on lines which have measurements for wind, sun, etc. -
  considering modern data and modern sensors only might generate a more accurate model; at very least worth exploring
  '''
  # General data cleaning (read, filter, rename)
  dat_init = pd.read_csv(weatherCSV, index_col = 'DATE')
  dat_filtered = dat_init[['PRCP','TMAX','TMIN','SNOW','SNWD']]
  dat_filtered.columns = ['precipitation', 'max_temp', 'min_temp', 'snowfall', 'snowdepth']
  dat_filtered.index = pd.to_datetime(dat_filtered.index)

  # Back fill min/max temp, and remove any columns with > 10% NA, otherwise drop rows with NA values
  zero_values = dat_filtered.apply(pd.isnull).sum()
  zero_percentage = zero_values / dat_filtered.apply(len)
  if zero_values['max_temp'] > 0 or zero_values['min_temp'] > 0: 
    dat_filtered[['max_temp','min_temp']].ffill(inplace = True)
  for col in dat_filtered.columns:
    if zero_percentage[col] > 0.1: 
      dat_filtered.drop(col)
  dat_filtered = dat_filtered.dropna()

  # Implement more predictors via data transformation
  dat_filtered['30_day_max'] = dat_filtered['max_temp'].rolling(30).mean()
  dat_filtered['month_day_max'] = dat_filtered['30_day_max'] / dat_filtered['max_temp']
  dat_filtered['max_min_ratio'] = dat_filtered['max_temp'] / (dat_filtered['min_temp'] + 0.00001) # avoid dividing by 0
  dat_filtered['monthly_avg'] = dat_filtered['max_temp'].groupby(dat_filtered.index.month, group_keys=False).apply(lambda x: x.expanding(1).mean())
  dat_filtered['day_of_year_avg'] = dat_filtered['max_temp'].groupby(dat_filtered.index.day_of_year, group_keys=False).apply(lambda x: x.expanding(1).mean())
  dat_filtered['next_day_max'] = dat_filtered.shift(-1)['max_temp'] # Target
  dat_filtered['next_week_max'] = dat_filtered.shift(-7)['max_temp'] # Target
  dat_filtered = dat_filtered.dropna()

  print(f'{len(dat_init)} original data points reduced to {len(dat_filtered)} data points after processing')
  return dat_filtered

processed = process_data('central_park.csv')

def build_ridge(data, target):
  predictors = data.columns.drop(['next_day_max','next_week_max'])

  model = Ridge(alpha = 0.1)
  train = data.loc[:data.index[round(len(data)*0.8)]]
  test = data.loc[data.index[round(len(data)*0.8)]:]
  model.fit(train[predictors], train[target])

  prediction = model.predict(test[predictors])
  error = mean_absolute_error(test[target], prediction)
  comparison = pd.concat([test[target], pd.Series(prediction, index=test.index)], axis = 1)
  comparison.columns = ['Actual_Temperature', 'Predicted_Temperature']
  return model, error, comparison

ridge_model_week, ridge_error_week, ridge_compare_week = build_ridge(processed, 'next_week_max')
ridge_model_day, ridge_error_day, ridge_compare_day = build_ridge(processed, 'next_day_max')