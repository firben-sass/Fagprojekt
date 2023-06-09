import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class FdaPlot:
    def __init__(self, S_data, time_df, s=1, a=0.8, n_sample = 10):
        self.S_data = S_data
        self.time_df = time_df
        self.s = s
        self.a = a
        self.n_sample = n_sample
    
    def plot(self):
        df_plot = pd.DataFrame(self.S_data.T)
        df_plot_n = df_plot.iloc[:, :self.n_sample]
        df_plot_n['time'] = self.time_df
        df_plot_melt_n = pd.melt(df_plot_n, id_vars='time')
        plt.figure()
        plot_n = sns.lineplot(data=df_plot_melt_n, x='time', y='value', hue='variable', 
                            linewidth=self.s, alpha=self.a, palette='Set1', legend=False)
        plot_n.set(xlabel='Integers relating to different wave numbers', ylabel='Absorbance of wine samples', title='')
        plt.show()



