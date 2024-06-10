# Source Link: https://www.youtube.com/playlist?list=PLjVLYmrlmjGfhqSO3rF4n02rrj9w2Ch2C

# Seaborn plot functions:
'''
lineplot()
displot() >> Histogram Plot
barplot()
scatterplot()
heatmap()
countplot()
violinplot()
pairplot()
stripplot()
boxplot()
catplot()
'''

# Line Plot with it's perameters
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

a = [1,2,3,4,5,6,7,8,9]
b = [9,8,7,6,5,4,3,2,1]
c = pd.DataFrame({'a':a, 'b':b})

sns.lineplot(x='a', y='b', data=c )

plt.show()

# Line Plot using gitHub Dataset
#https://github.com/mwaskom/seaborn-data
ghd = sns.load_dataset('penguins').head(100)
ghd

sns.lineplot(
             x='bill_length_mm',
             y='bill_depth_mm', 
             data=ghd, 
             hue='sex', 
             size=100, 
             style='sex', 
             palette='Accent', 
             markers=['*', 'o'],
             dashes=True,
             legend='full'
            )

plt.grid()
plt.title('Penguins', fontsize=15)

plt.show()

# Histogram Plot with it's perameters with GitHub Dataset
sns.displot(
            df['bill_length_mm'], 
            bins=[34,36,38,40,42,44,46], 
            kde=True, 
            rug=True,
            color='brown',
           ) # log_scale=True
plt.show()

# Bar Plot with GitHub Dataset
sns.barplot(x=df.island, y=df.bill_length_mm)
plt.show()

# Bar Plot with it's perameters with GitHub Dataset
sns.set(style='darkgrid')
sns.barplot(
            x='island', 
            y='bill_length_mm', 
            data=df, 
            hue='sex',
            order=['Dream', 'Torgersen', 'Biscoe'],
            hue_order=['Female', 'Male'],
            #ci=100
            errorbar=('ci', 100),
            n_boot=2,
            orient='v',
            #color='r' 
            palette='icefire',
            saturation=50,
            errcolor='r',
            errwidth=3,
            capsize=0.2,
            dodge=True,
            alpha=1
           )
plt.show()

# Scatter Plot with it's perameters with GitHub Dataset
sns.scatterplot(
                x='bill_length_mm', 
                y='bill_depth_mm', 
                data=df,
                hue='sex',
                style='sex',
                size='sex',
                sizes=(50, 100),
                palette='icefire',
                alpha=1,
                markers={'Male':'*', 'Female':'o'}
               )
plt.show()

# HeatMap Plot with it's perameters with GitHub Dataset
w = np.linspace(1,10,20).reshape(4,5)
w

v = sns.heatmap(w,
            vmin=0, 
            vmax=15, 
            cmap='gist_heat', 
            annot=True, # annot=df >> use for 'df' variable 
            annot_kws={'fontsize':10, 'color':'white'},
            #fmt='s' >> this is use for string (field name)
            linewidth=5,
            linecolor='b',
            cbar=True,
            xticklabels=True,
            yticklabels=True,
           )

v.set(xlabel='mona', ylabel='lisa')
sns.set(font_scale=0.5)

plt.show()
#____________________________________________________________________
a = df.drop(columns=['species', 'island', 'sex'], axis=1).head(10)
a

sns.heatmap(a)
plt.show()

# Count Plot with it's perameters with GitHub Dataset
sns.countplot(
                x='sex', 
                data=df, 
                hue='island', 
                #palette='bwr',
                #color='r',
                #saturation=0.9
             )
plt.show()

# Violin Plot with it's perameters with GitHub Dataset
sns.violinplot(
                x='island', 
                y='bill_length_mm', 
                data=df, 
                hue='sex', 
                linewidth=2, 
                palette='Dark2',
                order=['Dream', 'Torgersen', 'Biscoe'],
                #saturation=0.2,
                #color='r',
                split=True,
                scale='count', # 'area', 'width',
                #inner='quart', # {'point','stick','box','quartile',None}
                
              )
plt.show()
#______________________________________________________________
sns.violinplot(x=df['body_mass_g'])
plt.show()

# Pair Plot with it's perameters with GitHub Dataset
sns.pairplot(
                df, 
                #vars=['bill_length_mm','bill_depth_mm'],
                hue='sex',
                hue_order=['Female', 'Male'],
                markers=['o','>']
                #palette='BuGn'
                #x_vars=['bill_length_mm','bill_depth_mm'],
                #kind='kde', # 'kde','hist','scatter','reg'
                #diag_kind='hist'
            )
plt.show()

# Strip Plot with it's perameters with GitHub Dataset
sns.stripplot(
                x='island', 
                y='bill_length_mm', 
                data=df, 
                hue='sex', 
                palette='rocket',
                linewidth=1,
                edgecolor='black',
                jitter=2,
                size=5,
                marker='^',
                alpha=0.9
             )
plt.show()
#___________________________________________________
sns.stripplot(x=df['bill_length_mm'])
plt.show()

# Box Plot with it's perameters with GitHub Dataset
sns.set(style='whitegrid')
sns.boxplot( 
            x='island',
            y='bill_length_mm', 
            data=df,
            hue='sex',
            #color='r',
            order=['Dream', 'Torgersen', 'Biscoe'],
            showmeans=True,
            meanprops={'marker':'*', 'markeredgecolor':'white'},
            linewidth=1,
            palette='plasma',
            #orient='v'
           )
plt.show()
#__________________________________________________________
sns.set(style='whitegrid')
sns.boxplot( 
            data=df,
            showmeans=True,
            meanprops={'marker':'*', 'markeredgecolor':'white'},
            linewidth=1,
            palette='plasma',
            orient='h'
           )
plt.show()

# Cat Plot with it's perameters with GitHub Dataset
#df = pd.read_csv('penguins')
sns.catplot(
            x='bill_length_mm', 
            y='bill_depth_mm', 
            data=df, 
            hue='sex',
            palette='RdPu',
            height=5,
            kind='point'
           )
# factorplot >> catplot
# Category: 1) Scatter Plot. 2) Distribution Plot. 3) Estimate Plot
#1) Scatter Plot: 1) stripplot() with kind='strip'. 2) swarmplot() with kind='swarm'
#2) Distribution Plot: 1) boxplot() with kind='box'. 2) violinplot() with kind='violin'. 3) boxenplot() with kind='boxen'
#3) Estimate Plot: 1) pointplot() with kind='point'. 2) barplot() with kind='bar'. 3) countplot() with kind='count'
plt.show()

# Style Plot with it's perameters with GitHub Dataset
#1) Figue Style
#2) Axes Spines
#3) Scale
#4) Context
sns.set_style('darkgrid')
sns.set_context('paper', font_scale=2) #'poster'
plt.figure(figsize=(10,5))
sns.boxplot(x='island', y='bill_length_mm', data=df, palette='cool')
sns.despine()
plt.show()

# FacetGrid Plot with it's perameters with GitHub Dataset
fg = sns.FacetGrid(df, col='island', hue='sex', palette='cool')
fg.map(plt.scatter, 'bill_length_mm', 'bill_depth_mm', edgecolor='b').add_legend()
plt.show()