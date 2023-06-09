import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects as robjects
from load_data import load_data


"""
Plotting inspiration from: 
https://ranibasna.github.io/ddk/articles/DKK_and_Functional_analysis_on_wine_data.html
"""

#Note that the data is not standardized

x_learning, x_test = load_data()

def df_plot_fda(S_data, time_df, s=1, a=0.8, n_sample=10) -> None:
    
    df_plot = pd.DataFrame(S_data.T)
    df_plot_n = df_plot.iloc[:, :n_sample]
    df_plot_n['time'] = time_df
    df_plot_melt_n = pd.melt(df_plot_n, id_vars='time')
    plt.figure()
    plot_n = sns.lineplot(data=df_plot_melt_n, x='time', y='value', hue='variable', 
                          linewidth=s, alpha=a, palette='Set1', legend=False)
    plot_n.set(xlabel='Integers relating to different wave numbers', ylabel='Absorbance of wine samples', title='')
    plt.savefig("Wine_data_df_plot.png", dpi=800)
    return plot_n

t_df_wine = np.arange(1, x_learning.shape[1]+1)
plot_n = df_plot_fda(S_data=x_learning, time_df=t_df_wine, a=1, s=0.5, n_sample=10)


print("There are a total of", len(x_learning)+len(x_test), "samples in the entire dataset.")
print("There are a total of", len(x_learning), "samples in the training set.")
print("There are a total of", len(x_test), "samples in the test set.")