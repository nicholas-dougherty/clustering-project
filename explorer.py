def plot_continuous_duo(df, x, y):
    '''
        Create line and scatter plots along with a regression line for two 
        continuous variables. User provides a Pandas DataFrame and strings
        capturing the column names to be used for the independent variable, x,
        and dependent variable, y.  
    '''

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (4, 12))
    mean = df[y].mean()

    sns.lineplot(data = df, x = x, y = y, ax = ax[0])
    ax[0].axhline(mean, ls='--', color='grey')

    sns.scatterplot(data = df, x = x, y = y, ax = ax[1], alpha = 0.3, color = 'blue')
    ax[1].axhline(mean, ls='--', color='grey')

    plt.show()
