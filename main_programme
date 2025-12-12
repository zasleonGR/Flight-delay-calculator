import pandas as pnd
import numpy as np
import requests, zipfile, io
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Programme is loading.... (loading times may vary depending on your device)")

#load the compressed csv file from my github repository
url = "https://raw.githubusercontent.com/zasleonGR/Flight-delay-calculator/main/smaller.csv.zip"
response = requests.get(url)
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    with z.open("smaller.csv") as f:
        DataFrame = pnd.read_csv(f)

DataFrame = DataFrame[DataFrame.iloc[:, 3] != True] #delete the cancelled flights (rows) from the data, so that it does not mess up our predictions
DataFrame = DataFrame.replace(np.nan, 0.0) #replace all NaN values with 0

x = DataFrame[['DayofMonth', 'Month', 'Airline', 'Origin', 'Dest', 'CRSDepTime']]
y = DataFrame['DepDelayMinutes']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

#replacing string variables with a corresponding integer
label_encoder = LabelEncoder()
x_train_label = x_train.copy()
x_test_label = x_test.copy()
x_temp = x.copy()
x_temp['Airline'] = label_encoder.fit_transform(x_temp['Airline'])
x_temp['Origin'] = label_encoder.fit_transform(x_temp['Origin'])
x_temp['Dest'] = label_encoder.fit_transform(x_temp['Dest'])
x_train_label['Airline'] = label_encoder.fit_transform(x_train_label['Airline'])
x_train_label['Origin'] = label_encoder.fit_transform(x_train_label['Origin'])
x_train_label['Dest'] = label_encoder.fit_transform(x_train_label['Dest'])
x_test_label['Airline'] = label_encoder.fit_transform(x_test_label['Airline'])
x_test_label['Origin'] = label_encoder.fit_transform(x_test_label['Origin'])
x_test_label['Dest'] = label_encoder.fit_transform(x_test_label['Dest'])


mlr = LinearRegression()  
mlr.fit(x_train_label, y_train)

print("Welcome user to the Airline Delay Calculator!")
print("For prediction of test model set and model evaluation please insert 1, if you want to skip to the programme insert any other button")
prechoice = input()
if prechoice == '1':
    print("Intercept: ", mlr.intercept_)
    print("Coefficients:",list(zip(x, mlr.coef_)))

    #Prediction of test set
    y_pred_mlr= mlr.predict(x_test_label)
    #Predicted values
    print("Prediction for test set: {}".format(y_pred_mlr))

    #Actual value and the predicted value
    slr_diff = pnd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})

    #Model Evaluation
    meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
    meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))
    print('R squared: {:.2f}'.format(mlr.score(x_temp,y)*100))
    print('Mean Absolute Error:', meanAbErr)
    print('Mean Square Error:', meanSqErr)
    print('Root Mean Square Error:', rootMeanSqErr)

temp = DataFrame['Origin']
set1 = set(temp) #set is used so that all location are printed once
print("Please select one of the following origin locations:")
print(set1)
while True: #loop is used to prevent wrong input from the user
    origin_sel = input("origin selection (e.g.: ABY): ")
    if origin_sel not in set1:
        print("ERROR: origin location does not exist. Please try again.")
        print(" ")
    else:
        break

print(" ")
print("From the same list please choose a destination (WARNING: destination must be different from the origin chosen)")
while True:
    destination_sel = input("destination selection (e.g.: ATL): ")
    if destination_sel == origin_sel:
        print("ERROR: destination and origin locations are the same. Please try again.")
        print(" ")
    elif destination_sel not in set1:
        print("ERROR: destination location does not exist. Please try again.")
        print(" ")
    else:
        #create dictionaries so that each string variable matches their corresponding encoded integer
        temp = DataFrame['Origin']
        temp2 = x_temp['Origin']
        dict2 = {}
        for A, B in zip(temp, temp2):
            dict2[A] = B

        temp = DataFrame['Dest']
        temp2 = x_temp['Dest']
        dict3 = {}
        for A, B in zip(temp, temp2):
            dict3[A] = B

        enc_ori = dict2[origin_sel]
        enc_des = dict3[destination_sel]

        filt_array = x_temp[(x_temp.iloc[:, 3] == enc_ori) & (x_temp.iloc[:, 4] == enc_des)] #save only the rows with the origin and destination given
        if filt_array.empty: #check if array is empty from non existent flights
            print("ERROR: no available flights for your origin to that destination. Please select another location.")
            print(" ")
        else:
            break
print(" ")

temp = mlr.predict(filt_array)
final_array = x[(x.iloc[:, 3] == origin_sel) & (x.iloc[:, 4] == destination_sel)]
final_array.loc[:, 'Approx_delay'] = temp
print(final_array.to_string(index=False)) #".to_string" stops pandas from printing row numbers
print(" ")

big_delay = final_array[final_array.iloc[:, 6] > 15.0]
if big_delay.empty:
    print("Your flight path will not have any long delays all year round")
else:
    print("WARNING: long delay predictions detected!")
    print(big_delay.to_string(index=False))
