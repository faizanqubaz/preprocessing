import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder

def encodCategoricalColumns(df):
    print(df)
    ohe=OneHotEncoder(drop='first', sparse_output=False)
    sc=OrdinalEncoder(categories=[['Primary School','High School','Graduate','Masters','Phd']])
    encoded_data=ohe.fit_transform(df[['gender','relevent_experience','enrolled_university','major_discipline']])
    od_encoding = sc.fit_transform(df[['education_level']])
    feature_names1 = ohe.get_feature_names_out(['gender', 'relevent_experience', 'enrolled_university', 'major_discipline'])
    feature_names2=sc.get_feature_names_out(['education_level'])
    feature_names = np.concatenate((feature_names1,feature_names2))
    encoded_data_array1 = np.array(encoded_data)
    encoded_data_array2 = np.array(od_encoding)
    encoded = np.concatenate((encoded_data_array1,encoded_data_array2),axis=1)
    combined_data=pd.DataFrame(encoded, columns=feature_names)


    return {
        "X_train_new": combined_data,
        "feature_names": feature_names

    }