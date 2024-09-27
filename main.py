#this project is to predict rain on a specific month in the future for the city of southampton
import pandas as pd #we have imported the pandas package and saved it as 'pd'
from sklearn.linear_model import LinearRegression # so we can make future predictions using past data
from sklearn.model_selection import train_test_split# to ensure that the right data is being learnt so we can get an outcome from past data
from sklearn.metrics import mean_squared_error

# we have a DataFrame object named data which reads the csv file, MET_office data found online
data: pd.DataFrame = pd.read_csv('MET_office_weather_data.csv')
new_data = data.dropna() #since the data online had some NaN values, we use the dropna() method
# had an error because I wasn't aware of the nan values

#relevant data (location , date , time, rain)
#filter rows for relevant station
filtered_data = new_data[new_data['station'] == 'southampton']
features: list[str] = ['year','month']
target: str = 'rain'

# x = independent variables = processing data, y = dependant variables =  output
x: pd.DataFrame = filtered_data[features]
y: pd.tseries = filtered_data[target]

x.to_csv('X.csv',index=False) # writing the x object onto a csv
y.to_csv ('series_data.csv', header=True, index=False)

# Split the data into training and testing sets to learn data and get an outcome, training with 80% and testing with 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) # split arrays into train and test subsets

# Initializing and training the linear regression model
model = LinearRegression()

model.fit(x_train, y_train)  #training the model using x and y train objects to make future predictions

# Make predictions with a first test data processed
predictions = model.predict(x_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)  # Calculates the mean squared error (MSE) between the actual target variable (y_test)
# and the predicted values (predictions).
print("Mean Squared Error: {mse}")# depending on on % of our mse we can tell how accurate our prediciton is

# Forecasting: Predicting future rain based on new data
# Prompting user for input, while we have user input
while True:
    input_year = input("Enter the year: ")
    input_month = input("Enter the month: ")

    # display year and month chosen
    print("Year:", input_year)
    print("Month:", input_month)

    test_prediction_data = pd.DataFrame({#creates a DataFrame containing new data for prediction with features ('year', 'month', 'station') for two instances
        'year': [input_year],
        'month': [input_month],
    })

    # Use the trained model to predict rain for the new data
    new_predictions = model.predict(test_prediction_data)
    print("Predicted rain for new data:")
    print(new_predictions)












