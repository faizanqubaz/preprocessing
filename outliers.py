from matplotlib import pyplot as plt
import seaborn as sns

def checkandRemoveOutliers(df,ncols):
    checkOutliersGraph(df[ncols])
    data=detectOutliers(df[ncols])
    new_data=removeOutliers(data['lower'],data['higher'],df)
    data=checkOutliersGraph(new_data[ncols])
    return new_data
    


def checkOutliersGraph(df):
    for col in df.columns:
        plt.figure()
        plt.subplot(111)
        sns.boxplot(df[col])

        plt.show()

def detectOutliers(df):
    print(df)
    q1=df['city_development_index'].quantile(0.25)
    q3=df['city_development_index'].quantile(0.75)

    iqr=q3 - q1

    lower = q1 - 1.5 * iqr
    higher = q3 + 1.5 * iqr

    return {
        "lower":lower,
        "higher":higher
    }

def removeOutliers(lower,higher,df):
    outliers = df[(df['city_development_index']  < lower) | (df['city_development_index'] > higher)]

    new_data = df[(df['city_development_index'] > lower) & (df['city_development_index'] < higher)]
    return new_data
