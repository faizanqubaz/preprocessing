from sklearn.impute import SimpleImputer
import numpy as np


def imputenullValues(df):
    si=SimpleImputer(strategy='mean')
    si_mode=SimpleImputer(strategy='most_frequent')


    return {
        "si_city_dev": si.fit_transform(np.array(df['city_development_index']).reshape(-1,1)),
        "si_gender":si_mode.fit_transform(np.array(df['gender']).reshape(-1,1)),
        "si_enrolled_university":si_mode.fit_transform(np.array(df['enrolled_university']).reshape(-1,1)),
        "si_major_discipline":si_mode.fit_transform(np.array(df['major_discipline']).reshape(-1,1)),
        "si_experience":si.fit_transform(np.array(df['experience']).reshape(-1,1)),
        "si_company_size":si_mode.fit_transform(np.array(df['company_size']).reshape(-1,1)),
        'Education_l':si_mode.fit_transform(np.array(df['education_level']).reshape(-1,1))
    }
    