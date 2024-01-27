from sklearn.preprocessing import StandardScaler
import numpy as np


def standarizingData(df):
    st = StandardScaler()
    return {
        "city_index":st.fit_transform(np.array(df['city_development_index']).reshape(-1,1)),
        "exp":st.fit_transform(np.array(df['experience']).reshape(-1,1))
    }
    