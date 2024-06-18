import pandas as pd
df = pd.read_csv('weather_data.csv', parse_dates=['day'])
#print(df)
rows, cols = df.shape
print(df.head(2)) #print only first couple rows
print(df.tail(3)) #print only last three rows
print(df[2:5])
print(df.columns) #all columns
print(df.event)  #specific column
print(df[['event', 'day']])
print("Max temperature is", df['temperature'].max()) # can also do .mean(), .min(), .std(), way more
print(df.describe())
print(df[df.temperature >= 32]) # can also do things like df.temperature = df.temperature.max()
print(df['day'][df.temperature >= 32]) # prints the day when temperature was greater than 32
print(df.index)