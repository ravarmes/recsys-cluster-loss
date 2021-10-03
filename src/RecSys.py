import pandas as pd
import RecSysALS
import RecSysKNN
import RecSysNMF
from RecSysExampleData20Items import RecSysExampleData20Items

class RecSys():
        
    def __init__(self, n_users, n_movies, top_users, top_movies, l, theta, k):
        self.n_users = n_users
        self.n_movies = n_movies
        self.top_users = top_users
        self.top_movies = top_movies
        self.l = l
        self.theta = theta
        self.k = k

    ###################################################################################################################
    # function to read the data
    def read_movielens_1M(self, n_users, n_movies, top_users, top_movies, data_dir):
        # get ratings
        df = pd.read_table('{}/ratings.dat'.format(data_dir),names=['UserID','MovieID','Rating','Timestamp'], sep='::', engine='python')

        # create a dataframe with movie IDs on the rows and user IDs on the columns
        ratings = df.pivot(index='MovieID', columns='UserID', values='Rating')
        
        movies = pd.read_table('{}/movies.dat'.format(data_dir), names=['MovieID', 'Title', 'Genres'], sep='::', engine='python')
                            
        user_info = pd.read_table('{}/users.dat'.format(data_dir), names=['UserID','Gender','Age','Occupation','Zip-code'], sep='::', engine='python')
        user_info = user_info.rename(index=user_info['UserID'])[['Gender','Age','Occupation','Zip-code']]
        
        # put movie titles as index on rows
        movieSeries = pd.Series(list(movies['Title']), index=movies['MovieID'])
        ratings = ratings.rename(index=movieSeries)
        
        # read movie genres
        movie_genres = pd.Series(list(movies['Genres']),index=movies['Title'])
        movie_genres = movie_genres.apply(lambda s:s.split('|'))

        if top_movies:
            # select the top n_movies with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=1) # quantitative ratings for each movie: Movie 1: 4, Movie 2: 5, Movie 3: 2 ...
            rows = num_ratings.nlargest(n_movies) # quantitative ratings for each movie (n_movies) sorted: Movie 7: 6, Movie 2: 5, Movie 1: 4 ...
            ratings = ratings.loc[rows.index] # matrix[n_movies rows , original columns]; before [original rows x original columns]
        
        if top_users:
            # select the top n_users with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=0) # quantitative ratings made by each user: User 1: 5, User 2: 5, User 3: 5, ...
            cols = num_ratings.nlargest(n_users) # quantitative evaluations by each user (n_users) sorted: User 1: 5, User 2: 5, User 3: 5, ...
            ratings = ratings[cols.index] # matrix [n_movies rows , original columns]; before [n_movies rows , original columns] (just updated the index)
        else:
            # select the first n_users from the matrix
            cols = ratings.columns[0:n_users]
            ratings = ratings[cols] # matrix [n_movies rows , n_users columns]; before [n_movies rows , original columns]

        ratings = ratings.T # transposed: matrix [n_users rows x n_movies columns];

        return ratings, movie_genres, user_info
    

    ###################################################################################################################
    # function to read the data
    def read_movielens_small(self, n_users, n_movies, top_users, top_movies, data_dir):
        # get ratings
        df = pd.read_table('{}/ratings.dat'.format(data_dir),names=['UserID','MovieID','Rating','Timestamp'], sep='::', engine='python')

        # create a dataframe with movie IDs on the rows and user IDs on the columns
        ratings = df.pivot(index='MovieID', columns='UserID', values='Rating')
        
        movies = pd.read_table('{}/movies.dat'.format(data_dir), names=['MovieID', 'Title', 'Genres', 'Price'], sep='::', engine='python')
                            
        user_info = pd.read_table('{}/users.dat'.format(data_dir), names=['UserID','Gender','Age','NA','SPI', 'MA', 'MR'], sep='::', engine='python')
        user_info = user_info.rename(index=user_info['UserID'])[['Gender','Age','NA','SPI', 'MA', 'MR']]
        
        # put movie titles as index on rows
        movieSeries = pd.Series(list(movies['Title']), index=movies['MovieID'])
        ratings = ratings.rename(index=movieSeries)
        
        # read movie genres
        movie_genres = pd.Series(list(movies['Genres']),index=movies['Title'])
        movie_genres = movie_genres.apply(lambda s:s.split('|'))

        if top_movies:
            # select the top n_movies with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=1) # quantitative ratings for each movie: Movie 1: 4, Movie 2: 5, Movie 3: 2 ...
            rows = num_ratings.nlargest(n_movies) # quantitative ratings for each movie (n_movies) sorted: Movie 7: 6, Movie 2: 5, Movie 1: 4 ...
            ratings = ratings.loc[rows.index] # matrix[n_movies rows , original columns]; before [original rows x original columns]
        
        if top_users:
            # select the top n_users with the highest number of ratings
            num_ratings = (~ratings.isnull()).sum(axis=0) # quantitative ratings made by each user: User 1: 5, User 2: 5, User 3: 5, ...
            cols = num_ratings.nlargest(n_users) # quantitative evaluations by each user (n_users) sorted: User 1: 5, User 2: 5, User 3: 5, ...
            ratings = ratings[cols.index] # matrix [n_movies rows , original columns]; before [n_movies rows , original columns] (just updated the index)
        else:
            # select the first n_users from the matrix
            cols = ratings.columns[0:n_users]
            ratings = ratings[cols] # matrix [n_movies rows , n_users columns]; before [n_movies rows , original columns]

        ratings = ratings.T # transposed: matrix [n_users rows x n_movies columns];

        return ratings, movie_genres, user_info

    ###################################################################################################################
    # compute_X_est: 
    def  compute_X_est(self, X, algorithm='RecSysALS', data_dir="Data/MovieLens-Small"):
        if(algorithm == 'RecSysALS'):
            
            # factorization parameters
            rank = 1 # before 20
            lambda_ = 1 # before 20 - ridge regularizer parameter

            # initiate a recommender system of type ALS (Alternating Least Squares)
            RS = RecSysALS.als_RecSysALS(rank,lambda_)

            X_est,error = RS.fit_model(X)

        elif(algorithm == 'RecSysKNN'):
            RecSysKNN
        elif(algorithm == 'RecSysNMF'):
            RecSysNMF
        elif(algorithm == 'RecSysExampleAntidoteData20Items'):
            RS = RecSysExampleData20Items()
            X_est, movie_genres, user_info = RecSysExampleData20Items.read_movieitems(40, 20, False, False, data_dir)
        else:
            RecSysNMF
        return X_est  
        

#######################################################################################################################