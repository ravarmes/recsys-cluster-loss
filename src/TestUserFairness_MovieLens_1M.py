from RecSys import RecSys
from UserFairness import Polarization
from UserFairness import IndividualLossVariance
from UserFairness import GroupLossVariance


# reading data from 3883 movies and 6040 users 
Data_path = 'Data/MovieLens-1M'
n_users=  300
n_movies= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_movies = False # True: to use movies with more ratings; False: otherwise

# recommendation algorithm
algorithm = 'RecSysALS'

# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_movies, top_users, top_movies, l, theta, k)

X, genres, user_info = recsys.read_movielens_1M(n_users, n_movies, top_users, top_movies, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_moveis columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

#print("\nuser_info")
#print(user_info)
#print("\nX")
#print(X)
#print("\nX_est")
#print(X_est)
#print("\nOmega")
#print(omega)

print("\n\n------------ SOCIAL OBJECTIVE FUNCTIONS ------------")

# To capture polarization, we seek to measure the extent to which the user ratings disagree
polarization = Polarization()
Rpol = polarization.evaluate(X_est)
print("Polarization (Rpol):", Rpol)

# Individual fairness. For each user i, the loss of user i, is  the mean squared estimation error over known ratings of user i
ilv = IndividualLossVariance(X, omega, 1) #axis = 1 (0 rows e 1 columns)
Rindv = ilv.evaluate(X_est)
print("Individual Loss Variance (Rindv):", Rindv)

# Group fairness. Let I be the set of all users/items and G = {G1 . . . ,Gg} be a partition of users/items into g groups. 
# The loss of group i as the mean squared estimation error over all known ratings in group i
# G group: identifying the groups (NA: users grouped by number of ratings for available items)
# advantaged group: 5% users with the highest number of item ratings
# disadvantaged group: 95% users with the lowest number of item ratings
list_users = X_est.index.tolist()
advantaged_group = list_users[0:15]
disadvantaged_group = list_users[15:300]
G = {1: advantaged_group, 2: disadvantaged_group}

glv = GroupLossVariance(X, omega, G, 1) #axis = 1 (0 rows e 1 columns)
RgrpNA = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NA):", RgrpNA)