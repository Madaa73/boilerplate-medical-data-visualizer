import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('/workspace/boilerplate-medical-data-visualizer/medical_examination.csv')

# 2
df['height'] = df['height'] / 100
df['overweight'] = (df['weight'] / (df['height'] ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df,id_vars=[ 'id','age','sex','height','weight','ap_hi','ap_lo','cardio'],value_vars=['cholesterol','gluc','smoke','alco','active','overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio','variable','value']).size().reset_index(name='total')
    #Renaming the variables of the melted df
    df_cat = df_cat.rename(columns={ 
        'variable': 'variable',
        'value': 'healthy'
    })

    # 7
    sns.catplot(
        data = df_cat,
        kind = 'bar',
        x = 'variable',
        y = 'total',
        hue = 'healthy',
        col = 'cardio',
        height = 5,
        aspect = 1
    )

    # 8
    fig = plt.gcf()
    plt.show()

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat =  df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  
        (df['height'] >= df['height'].quantile(0.025)) &  
        (df['height'] <= df['height'].quantile(0.975)) &  
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))  
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(10,8))

    # 15
    sns.heatmap(corr, mask = mask,annot=True, fmt=".1f", cmap='rocket', ax=ax)
    fig = plt.gcf()
    plt.show()

    # 16
    fig.savefig('heatmap.png')
    return fig

