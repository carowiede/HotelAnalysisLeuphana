import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as ss


def analyze_null_values(df):
    '''
    Function to analyze NULL-Values in a DataFrame
    :param df: DataFrame to analyze
    :return: variable with NULL-Values and the amount of NULL-Values or a message that no NULL-Values were found
    '''
    null_found = False  # initial variable to check if NULL-Values were found

    print('---- Variable und Anzahl NULL-Werte mit Anteil an Total ----')
    for column in df.columns:
        if df[column].isna().sum() != 0:
            null_count = df[column].isna().sum()
            null_percentage = (null_count / len(df.index)) * 100
            rounded_percentage = round(null_percentage, 2)
            print(column, ':', null_count, '(', rounded_percentage, '% )')
            null_found = True  # set variable to True if NULL-Values were found

    if not null_found:
        print('Keine NULL-Werte vorhanden')


def get_count(series, limit=None):
    '''
    INPUT:
        series: Pandas Series (Single Column from DataFrame)
        limit:  If value given, limit the output value to first limit samples.
    OUTPUT:
        x = Unique values
        y = Count of unique values in percent
    '''

    if limit != None:
        series = series.value_counts()[:limit]
    else:
        series = series.value_counts()

    x = series.index
    y = series / series.sum() * 100 # calculate percentage

    return x.values, y.values


def plot_values(x, y, x_label=None, y_label=None, title=None, figsize=(7, 5), plot_type='bar', palette='colorblind'):
    '''
    Function to plot data
    :param x: x values
    :param y: y values
    :param x_label: optional x label
    :param y_label: optional y label
    :param title: optional title
    :param figsize: optional figsize, default is (7,5)
    :param plot_type: optional plot_type, default is bar
    :return: plot
    '''
    sns.set_style('darkgrid')

    fig, ax = plt.subplots(
        figsize=figsize)

    if x_label:
        ax.set_xlabel(x_label)

    if y_label:
        ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    if plot_type == 'bar':
        # Erstelle ein Balkendiagramm mit Beschriftungen für die Balken
        bars = sns.barplot(x=x, y=y, ax=ax)
        for bar, percentage in zip(bars.patches, y):
            height = bar.get_height()
            ax.annotate(f'{percentage:.2f}%', (bar.get_x() + bar.get_width() / 2, height),
                        ha='center', va='bottom')  # rounded to 2 decimal places and show percantage above bar
    elif plot_type == 'line':
        sns.lineplot(x=x, y=y, ax=ax)


# Function from https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def cramers_v(x, y):
    '''
    '''
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def iqr_outlier(df, column):
    '''
    Function to detect outliers with the IQR-Method
    :param df:
    :param column:
    :return:
    '''
    median = df[column].median()
    max = df[column].max()
    min = df[column].min()

    # Ermittlung der Quantile
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = float(Q3 - Q1)
    lower_bound = float(Q1 - 1.5 * IQR)
    upper_bound = float(Q3 + 1.5 * IQR)
    print('Unteres Quantil:', Q1, ', Median: ', median, ', Oberes Quantil:', Q3, ', IQR: ', IQR)
    print('IQR Untere Grenze: ' , lower_bound, ', IQR Obere Grenze: ', upper_bound)
    IQR_out = df[column][((df[column] < lower_bound)
                                   |(df[column] > upper_bound))]
    print('Anzahl der Ausreißer: ', len(IQR_out), ', Anteil der Ausreißer: ', round(len(IQR_out)/len(df[column])*100, 2), '%')
    print('Minimum:', min, ', Maximum:', max)


def boxplot_hotel(df_city_hotel, df_resort_hotel, column, limit=True):
    '''
    Function to plot boxplots for each hotel type for a different columns
    :param df_city_hotel:
    :param df_resort_hotel:
    :param column:
    :param limit:
    :return:
    '''
    # optional: Berechnen Sie die gemeinsamen Achsenlimits für beide Diagramme
    min_limit = min(df_city_hotel[column].min(), df_resort_hotel[column].min())
    max_limit = max(df_city_hotel[column].max(), df_resort_hotel[column].max())

    # Erstellen Sie separate Boxplots für jeden Hoteltyp
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)  # Erster Subplot für City Hotel
    sns.boxplot(data=df_city_hotel, y=column)
    plt.xlabel('City Hotel')
    plt.ylabel(column)
    plt.title('Boxplot von ' + column + ' für City Hotel')
    if limit:
            plt.ylim(min_limit, max_limit)  # Setzen Sie die Achsenlimits

    # Zweiter Subplot für Resort Hotel
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_resort_hotel, y=column)
    plt.xlabel('Resort Hotel')
    plt.ylabel(column)
    plt.title('Boxplot von ' + column + ' für Resort Hotel')
    if limit:
        plt.ylim(min_limit, max_limit)  # Setzen Sie die Achsenlimits



# Definiere die Funktion zur Berechnung des upper_bound
def calculate_upper_bound(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = float(Q3 - Q1)
    upper_bound = float(Q3 + 1.5 * IQR)
    return upper_bound


def plot_conf(conf,model,cmap='Blues_r'):
    plt.figure(figsize=(5,3))
    sns.heatmap(conf, annot=True, fmt='d', cmap=cmap,
                xticklabels=['0', '1'],
                yticklabels=['0', '1'])

    plt.xlabel('Vorhergesagte Werte')
    plt.ylabel('Tatsächliche Werte')
    plt.title(model)


def plot_feature_importances(model, headline):
    '''
    Function to plot the feature importances of a model
    :param model: classification model
    :param headline: part of the title
    :return: plot
    '''
    headline = headline
    n_features = X_train.shape[1]
    feature_importances = model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::1]  # Sortiere die Indizes der Größe nach absteigend
    sorted_feature_importances = feature_importances[sorted_indices]
    sorted_feature_names = X_train.columns.values[sorted_indices]

    plt.figure(figsize=(8, 6))
    plt.barh(range(n_features), sorted_feature_importances, align='center')
    plt.yticks(np.arange(n_features), sorted_feature_names)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.title('Feature importances: ' + str(headline))
    plt.show()


