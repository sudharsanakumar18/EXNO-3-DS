## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
DEVELOPED BY: SUDHARSANA KUMAR S R
REGISTER NO.:212223240162
```
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data (1).csv")
df
```
![Screenshot 2024-03-26 161741](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/c037e654-423b-4490-9eb9-a3d95e8d13cd)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-03-26 161914](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/d803372f-942b-490f-990a-ad36c54de2ad)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-03-26 161955](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/03f5ad08-dcd6-4b33-b552-c1d3790f1139)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-03-26 162026](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/4f0921b4-f25c-42ec-a293-9a47e03d46d6)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
![Screenshot 2024-03-26 162107](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/dc85fd3c-a18c-420a-a164-ad4eb2aaba1b)

```
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-03-26 162139](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/aa1594a4-c6b5-48b5-ae8a-f898a499e65f)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2024-03-26 162202](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/f3d37cf0-8738-4512-bb35-9c0ab1916bad)

```
pip install --upgrade category_encoders
```
![Screenshot 2024-03-26 162234](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/4a980486-f04a-4083-b912-e98b8b4d7af5)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![Screenshot 2024-03-26 162301](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/febd56a0-ff6d-45ba-8a41-54764679b750)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-03-26 162328](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/ef17f6b8-efb0-4d43-87ac-57427978f05e)

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc['City'],y=cc['Target'])
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2024-03-26 162353](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/ee0ec54d-cf89-4470-9efb-af94d9a3df71)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-03-26 162420](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/c96128f7-f4a5-4740-a2e3-e78e0538ba7c)

```
df.skew()
```
![Screenshot 2024-03-26 162440](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/90fee750-9021-4e3c-b9c3-97e76b862e95)
```
np.log(df['Highly Positive Skew'])
```
![Screenshot 2024-03-26 220900](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/f1386e36-d4a9-40d2-b68d-7b3b738822d8)
```
np.reciprocal(df['Moderate Positive Skew'])
```
 ![Screenshot 2024-03-26 220941](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/a29c01bf-cf78-420f-bf77-db10934e0427)
```
np.sqrt(df['Highly Positive Skew'])
```
![Screenshot 2024-03-26 221050](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/b977d6b3-ac32-43fb-9bdd-de83e10500ad)
```
np.square(df['Highly Positive Skew'])
```
![Screenshot 2024-03-26 221121](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/7dfd945d-a7a2-488f-948f-608e1127ad28)
```
df['Highly Positive Skew_boxcox'],parameters=stats.boxcox(df['Highly Positive Skew'])
df
```
![Screenshot 2024-03-26 221200](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/6468acc7-56ef-4f7b-8c3b-a0cf820f8268)
```
df['Moderate Negative Skew_yeojohnson'],parameters=stats.yeojohnson(df['Moderate Negative Skew'])
df.skew()
```
![Screenshot 2024-03-26 221226](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/a488712a-5ccc-4cf6-a474-a24ecdd395fa)
```
df['Highly Negative Skew_yeojohnson'],parameters=stats.yeojohnson(df['Highly Negative Skew'])
df.skew()
```
![Screenshot 2024-03-26 221308](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/6abbb9d2-eeb2-4406-b5b0-6ae5b7dc30b3)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
```
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![Screenshot 2024-03-26 221345](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/22aa1cb7-cd9b-4b0e-a90a-6cb60fb4775d)
```
sm.qqplot(np.reciprocal(df['Moderate Negative Skew']),line='45')
plt.show()
```
![Screenshot 2024-03-26 221418](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/09c81b30-f0ed-41a4-8a05-6147ada9f4d1)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```
```
df['Moderate Negative Skew']=qt.fit_transform(df[['Moderate Negative Skew']])
```
```
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![Screenshot 2024-03-26 221520](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/d1c59de7-ec1a-4c5c-a7ca-2a9638ff5843)
```
df['Highly Negative Skew_1']=qt.fit_transform(df[['Highly Negative Skew']])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```
![Screenshot 2024-03-26 221609](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/22ec7075-48d8-4f5b-a2c0-28131b6e5fa2)
```
sm.qqplot(df['Highly Negative Skew_1'],line='45')
plt.show()
```
![Screenshot 2024-03-26 221636](https://github.com/Anusharonselva/EXNO-3-DS/assets/119405600/2d67db86-a44c-47f4-8205-b86e67de97ae)


# RESULT:
    Hence performing Feature Encoding and Transformation process is Successful.
