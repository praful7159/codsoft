import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = pd.read_csv('C:/Users/prafu/Downloads/advertising.csv')


print(data.head())


X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Root Mean Squared Error: {rmse}')


new_data = pd.DataFrame({
    'TV': [300],
    'Radio': [100],
    'Newspaper': [50]
})

predicted_sales = model.predict(new_data)
print(f'Predicted Sales: {predicted_sales[0]}')
