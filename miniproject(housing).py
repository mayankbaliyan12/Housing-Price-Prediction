HOUSING PRICE PREDICTION PROJECT USING MACHINE LEARNING
SUBMITTED TO: MANISH SHARMA 
SUBMITTED BY : 
MAYANK BALIYAN, CSE(DS), Roll No. 2013381
MRIDUL JOSHI, CSE(DS), Roll No. 2013386
PRABHAV SINGH NEGI, CSE(DS), Roll No. 2013417
#%%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mpl_toolkits
%matplotlib inline

# %%
#READING DATA
data = pd.read_csv("D:/study/data_science/mini project/house_data.csv")

# %%
data.head()

# %%
#DESCRIBING DATA
data.describe()

# %%
#BEDROOM VS COUNT
data['bedrooms'].value_counts().plot(kind='bar')
plt.title('NUMBER OF BEDROOMS')
plt.xlabel('BEDROOMS')
plt.ylabel('COUNT')
sns.despine

# %%
plt.figure(figsize=(10,10))
sns.jointplot(x=data.head(200).lat.values, y=data.head(200).long.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
sns.despine

# %%
#PRICE VS SQUARE FEET
plt.scatter(data.price,data.sqft_living)
plt.title("PRICE VS SQUARE FEET")

# %%
#PRICE VS LOCATION OF AREA
plt.scatter(data.price,data.long)
plt.title("PRICE VS LOCATION OF AREA")

# %%
#LATITUDE VS PRICE
plt.scatter(data.price,data.lat)
plt.xlabel("PRICE")
plt.ylabel('LATITUDE')
plt.title("LATITUDE VS PRICE")

# %%
#BEDROOM AND PRICE
plt.scatter(data.bedrooms,data.price)
plt.title("BEDROOM AND PRICE ")
plt.xlabel("BEDROOMS")
plt.ylabel("PRICE")
plt.show()
sns.despine

# %%
plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])

# %%
plt.scatter(data.waterfront,data.price)
plt.title("Waterfront vs Price ( 0= no waterfront)")

# %%
#CLEANING DATA
train1 = data.drop(['id', 'price'],axis=1)

# %%
train1.head()

# %%
data.floors.value_counts().plot(kind='bar')

# %%
plt.scatter(data.head(30).floors,data.head(30).price)

# %%
plt.scatter(data.condition,data.price)

# %%
plt.scatter(data.zipcode,data.price)
plt.title("Which is the pricey location by zipcode?")

# %%
#APPLYING REGRESSION
reg = LinearRegression()

# %%
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)

# %%
from sklearn.model_selection import train_test_split

# %%
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)

# %%
reg.fit(x_train,y_train)


# %%
reg.score(x_test,y_test)

# %%
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2, learning_rate = 0.1, loss = 'ls')

# %%
clf.fit(x_train, y_train)

# %%
clf.score(x_test,y_test)



# %%