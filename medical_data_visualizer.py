import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('./medical_examination.csv')

# Add 'overweight' column
# BMI = weight(kg)/height(m)^2. BMI > 25 is overweight
bmi = df['weight'].div(df['height'].div(100).pow(2))
# overweight value as boolean integer
df['overweight'] = (bmi > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df[['cholesterol', 'gluc']] = (df[['cholesterol', 'gluc']] > [1, 1]).astype(int)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars=['cardio'], value_vars =['active', 'alco','cholesterol', 'gluc', 'overweight', 'smoke'])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    # df_cat = ??

    # Draw the catplot with 'sns.catplot()'
    g = sns.catplot(x='variable', col='cardio', hue='value', kind='count', data=df_cat)
    g.set_ylabels('total')

    #! .catplot() returns a FacetGrid object; but, tests expect a Figure object
    fig = g.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    filter_ap = df.ap_lo.le(df.ap_hi) # Keep those where diastolic ap is less than or equal to systolic ap.
    filter_height = df.height.between(df.height.quantile(0.025), df.height.quantile(0.975)) #  Keep middle 95 percentiles (150.0 to 180.0)
    filter_weight = df.weight.between(df.weight.quantile(0.025), df.weight.quantile(0.975)) # Keep middle 95 percentiles (51.0 to 108.0) 

    df_heat = df[filter_ap & filter_height & filter_weight]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(
    corr, vmin=-0.15, vmax=0.3, cmap='icefire', center=0, annot=True, fmt='.1f', linewidths= 1, cbar_kws={'shrink': 0.5, 'ticks':[-0.08, 0.00, 0.08, 0.16, 0.24]}, square=True, mask=mask)


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
