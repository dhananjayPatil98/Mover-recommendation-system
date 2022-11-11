from typing import Collection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import warnings
import collections
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics_pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
data=pd.read_csv('netflix_titles.csv')
data.head()

net_movie = data.loc[data.type=='Movie',:].reset_index()
net_movie.title=net_movie.title.str.lower()
net_movie['index']=net_movie.index
net_movie.head()

tv_shows=data.loc[data['type']=='TV Show'].reset_index()
tv_shows.title=tv_shows.title.str.lower()
tv_shows['index']=tv_shows.index
tv_shows
'''
movie_data.duplicate().sum()
tv_shows.duplicated().sum()
'''
index=tv_shows.index
number_of_rows_tv=len(index)

index=net_movie.index
number_of_rows_movies=len(index)

color=['y','r']
label='TV Shows','Movies'
sizes=[number_of_rows_tv,number_of_rows_movies]
explode=(0.1,0)
fig1,ax1=plt.subplots()
ax1.pie(sizes,explode=explode,labels=label,colors=color,autopct='%2.2f%%',shadow=True,startangle=120)
ax1.axis('equal')
plt.show()

top_15=net_movie.sort_values(by='release_year',ascending=False).head(15)
top_15[["title","release_year"]]

top_country=net_movie['country'].value_counts().rename_axis('Country').reset_index(name='counts')[:10]
fig=px.bar(top_country,y='Country',x='counts',orientation='h',title='Country with the most number of titles', color="counts", color_continuous_scale=px.colors.qualitative.Prism).update_yaxes(categoryorder='total ascending')
fig.show()

newdata=net_movie
new=newdata.groupby("listed_in").count()
category=new.sort_values(by='index',ascending=False).head(10)
category1=category[["type"]]
category1.plot(kind="barh")

net_movie['time'] = net_movie['duration'].str.split(' ',expand = True)[0]

net_movie['time'] = net_movie['time'].astype(int)

net_movie['screenplay'] = net_movie['time']/60

top_20 = net_movie.sort_values(by = 'screenplay', ascending = False).head(20)

plt.figure(figsize = (12, 10))
sns.barplot(data = top_20, y = 'title', x = 'screenplay', hue = 'country', dodge = False)
plt.legend(loc = 'lower right')
plt.xlabel('Total Hours')
plt.ylabel('Movie')
plt.title('Top 20 movies by run time')
plt.show()