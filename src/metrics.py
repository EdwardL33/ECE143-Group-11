import pandas as pd


def print_genre_metrics_one_hot(df):
    '''
        prints the genre metrics for a dataset
    '''
    genre_dummies = df['track_genre'].str.get_dummies(sep=', ')

    # list to store calculations
    genre_metrics = []

    # loop through genres
    for genre in genre_dummies.columns:
        # if this genre is true for this song get the popularity
        is_genre_present = genre_dummies[genre] == 1
        genre_popularity = df.loc[is_genre_present, 'popularity']
        
        # get mean and count and save into list
        genre_metrics.append({
            'genre': genre,
            'mean': genre_popularity.mean(),
            'count': genre_popularity.count()
        })

    # convert list into dataframe
    genre_stats_df = pd.DataFrame(genre_metrics).set_index('genre')

    # print max rows
    print(genre_stats_df.sort_values(by='mean', ascending=False))

def print_genre_metrics(df):
    genre_stats = df.groupby('track_genre')['popularity'].agg(['mean', 'count'])

    # Filter out genres with fewer than, say, 50 tracks
    min_tracks = 50
    reliable_genres = genre_stats[genre_stats['count'] >= min_tracks]

    print(reliable_genres.sort_values(by='mean', ascending=False))

