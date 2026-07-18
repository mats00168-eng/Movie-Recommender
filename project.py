import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

df = pd.read_csv('sample_movies')
titles = df['title'].str.lower().tolist()
tdf = TfidfVectorizer().fit_transform(df['genres'])
sim = cosine_similarity(tdf)
def recommend(movie_name):
    movie_name = movie_name.lower().strip()
    if movie_name not in titles:
        close = get_close_matches(movie_name,titles,cutoff=0.6,n=1)
        if close:
            movie_name = close[0]
            print(f'Did you mean {movie_name}?')
        else:
            return 'Movie not found'
    movie_index = df[df['title'].str.lower() == movie_name].index[0]
    scores = list(enumerate(sim[movie_index]))
    scores = sorted(scores,key=lambda x: x[1],reverse=True)
    result = []
    for i in scores[1:6]:
        result.append(df['title'][i[0]])
    return result
        
print(2)