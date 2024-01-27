from matplotlib import pyplot as plt
import seaborn as sns

def checkDistribution(df):
    for col in df.columns:
        plt.figure()
        plt.subplot(111)
        sns.distplot(df[col])

        plt.show()
