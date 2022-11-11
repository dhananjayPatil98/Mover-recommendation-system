from rake_nltk import Rake
import pandas as pd
import numpy as npx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv('IMDB_Top250Engmovies2_OMDB_Detailed.csv')
df.head()
df['Director'].value_counts()[0:10].plot.bar('barh', figsize=[8,5], fontsize=15, color='navy')
df['Key_words'] = ''

r = Rake()
for index, row in df.iterrows():
    r.extract_keywords_from_text(row['Plot'])
    key_words_dict_scores = r.get_word_degrees()
    row['Key_words'] = list(key_words_dict_scores.keys())
df['Genre'] = df['Genre'].map(lambda x: x.split(','))
df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])
df['Director'] = df['Director'].map(lambda x: x.split(','))

for index, row in df.iterrows():
    row['Genre'] = [x.lower().replace(' ','') for x in row['Genre']]
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
    row['Director'] = [x.lower().replace(' ','') for x in row['Director']]
df['Bag_of_words'] = ''
columns = ['Genre', 'Director', 'Actors', 'Key_words']
for index, row in df.iterrows():
    words = ''
    for col in columns:
        words += ' '.join(row[col]) + ' '
    row['Bag_of_words'] = words
    
df = df[['Title','Bag_of_words']]
print(df['Bag_of_words'])
ls=df['Bag_of_words']

count = CountVectorizer()
count_matrix = count.fit_transform(ls)
#print(df)
'''cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)
indices = pd.Series(df['Title'])
def recommend(title, cosine_sim = cosine_sim):
    recommended_movies = []
    idx = indices[indices == title].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indices = list(score_series.iloc[1:11].index)
    
    for i in top_10_indices:
        recommended_movies.append(list(df['Title'])[i])
        
    return recommended_movies

recommend('The Avengers')'''