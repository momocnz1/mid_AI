from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

File_Path = 'C:/Users/User/Downloads/'
File_Name ='car_data.csv'

df = pd.read_csv(File_Path + File_Name)


df.drop(columns=['User ID'],inplace=True)
encoders = []
for i in range(0,len(df.columns)):
    en = LabelEncoder()
    df.iloc[:,i] = en.fit_transform(df.iloc[:,i])
    encoders.append(en)

x = df.iloc[:,0:3]
y = df['Purchased']

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=0)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)
x_pred = ['Male','35','20000']

for i in range(0,len(df.columns)):
         x_pred[i] = encoders[i].transform([x_pred[i]])
         x_pred_adj = np.arry(x_pred).reshape(-1,5)
        
y_pred = model.predict(x_pred_adj)
print('Prediiction:', y_pred[0])
score = model.score(x_train,y_train)
print('Accuracy:','{:.2f}'.format(score))

print(model.score(x_train, y_train))
print(model.score(x_test,y_test))
print('********************')
print(df.isnull().any())
print('********************')
print(df.isnull().sum())
df.dropna(inplace=True)
