import pandas as pd
from sklearn.model_selection import train_test_split
from distribution import checkDistribution
from impute import imputenullValues
import numpy as np
from outliers import checkandRemoveOutliers
from standard import standarizingData
from encode import encodCategoricalColumns

# READ THE DATA
data = pd.read_csv(r"C:/Users/CL/Desktop/data_science.csv")
print(data)
# FIRST CHECK IF ANY COLUMN HAS 100% NULL VALUES
null_columns = data.columns[data.isnull().all()]

if len(null_columns) !=0:
    data.drop(columns=null_columns)

X_train,X_test,Y_train,Y_test=train_test_split(data.iloc[:,2:10],data.iloc[:,-1],test_size=0.2,random_state=42)

# SPIT INTO NUMERICAL AND CATEGORICAL COLUMNS
numerical_columns = [col for col in X_train if X_train[col].dtypes == 'float64']
categorical_columns = [col for col in X_train if X_train[col].dtypes == 'object']

# checkDistribution(X_train[numerical_columns])

# IMPUTE NULL VALUES
imputer=imputenullValues(X_train)

# removing prev values from x_train
remaining_cols = X_train.drop(columns=['city_development_index','gender','enrolled_university','education_level','major_discipline','experience','company_size'])

combined_cols = np.concatenate([imputer['si_city_dev'],imputer['si_gender'],remaining_cols,imputer['si_enrolled_university'],imputer['Education_l'],imputer['si_major_discipline'],imputer['si_experience'],imputer['si_company_size']],axis=1)

new_data = pd.DataFrame(combined_cols,columns=X_train.columns)


# DETECT THE OUTLIERS
outlier_new_data=checkandRemoveOutliers(new_data,numerical_columns)

# LETS DO STANDARIZATION FIRST
sd_data=standarizingData(outlier_new_data[numerical_columns])

# Removing the experience and city developmentindex first
remaing_columns=outlier_new_data.drop(columns=['city_development_index','experience'])

combined_scalar_cols = np.concatenate([sd_data['city_index'],remaing_columns,sd_data['exp']],axis=1)

new_scalar_data = pd.DataFrame(combined_scalar_cols,columns=outlier_new_data.columns)


# ENCODING CATEGORICAL COLUMNS
encode_data=encodCategoricalColumns(new_scalar_data)
dd=new_scalar_data.drop(columns=['gender','relevent_experience','enrolled_university','major_discipline','education_level'])
combined_data = pd.concat([dd, pd.DataFrame(encode_data['X_train_new'], columns=encode_data['feature_names'])], axis=1)



