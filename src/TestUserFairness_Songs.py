from RecSys import RecSys
from UserFairness import Polarization
from UserFairness import IndividualLossVariance
from UserFairness import GroupLossVariance


# reading data from 19993 songs and 16000 users 
Data_path = 'Data/Songs'
n_users=  300
n_items= 1000
top_users = True # True: to use users with more ratings; False: otherwise
top_items = True # True: to use songs with more ratings; False: otherwise

# recommendation algorithm
algorithm = 'RecSysALS'

# parameters for calculating fairness measures
l = 5
theta = 3
k = 3

recsys = RecSys(n_users, n_items, top_users, top_items, l, theta, k)

X, genres, user_info = recsys.read_songs(n_users, n_items, top_users, top_items, data_dir = Data_path) # returns matrix of ratings with n_users rows and n_books columns
omega = ~X.isnull() # matrix X with True in cells with evaluations and False in cells not rated

X_est = recsys.compute_X_est(X, algorithm) # RecSysALS or RecSysKNN or RecSysNMF or RecSysExampleAntidoteData20Items

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
G1 = {1: advantaged_group, 2: disadvantaged_group}

glv = GroupLossVariance(X, omega, G1, 1) #axis = 1 (0 rows e 1 columns)
RgrpNA = glv.evaluate(X_est)
print("Group Loss Variance (Rgrp NA):", RgrpNA)


# G group: identifying the groups (IU: individual unfairness - the variance of the user losses)
# The configuration of groups was based in the hierarchical clustering (tree clustering - dendrogram)
# Clusters 1, 2 and 3
#G2 = {1: [882, 1167, 1178, 1184, 1214, 1249, 1279, 1376, 1435, 1436, 1558, 1596, 1674, 1706, 1898, 2012, 2036, 2106, 2139, 2222, 2255, 2288, 2295, 2313, 2385, 2406, 2589, 2766, 2977, 3145, 3167, 3363, 3371, 3373, 4017, 5027, 5037, 6073, 6251, 6563, 6575, 6772, 7158, 7283, 7286, 7841, 7915, 8066, 8454, 8681, 9908, 10314, 10447, 10819, 11224, 11676, 11788, 11944, 12489, 13518, 13552, 13850, 15418, 15602, 16246, 16599, 16634, 16795, 17003, 19233, 19493, 19664, 20106, 20462, 21014, 21031, 22074, 23547, 23902, 23933, 24194, 25410, 25601, 25966, 25981, 26535, 26621, 26883, 28177, 28204, 29209], 2: [6789, 15957, 21364, 26538], 3: [254, 638, 929, 1131, 1155, 1161, 1192, 1211, 1219, 1248, 1254, 1261, 1262, 1294, 1297, 1331, 1348, 1368, 1372, 1399, 1409, 1412, 1424, 1466, 1467, 1485, 1499, 1517, 1535, 1548, 1554, 1585, 1597, 1608, 1652, 1660, 1688, 1719, 1725, 1733, 1790, 1791, 1805, 1812, 1830, 1838, 1848, 1863, 1869, 1881, 1891, 1901, 1903, 1923, 1928, 1990, 2009, 2010, 2024, 2030, 2033, 2041, 2046, 2084, 2090, 2103, 2110, 2132, 2134, 2135, 2136, 2152, 2179, 2189, 2197, 2238, 2276, 2281, 2287, 2296, 2326, 2333, 2337, 2349, 2354, 2358, 2363, 2389, 2399, 2404, 2411, 2415, 2437, 2439, 2461, 2462, 2466, 2470, 2481, 2545, 2549, 2552, 2559, 2597, 2622, 2630, 2644, 2651, 2653, 2678, 2688, 2719, 2891, 3827, 4221, 4785, 4795, 4938, 5582, 5899, 5903, 6115, 6242, 6323, 6532, 6543, 7125, 7346, 8067, 8187, 8245, 8253, 8680, 8734, 8930, 9177, 9226, 9856, 10560, 11245, 11657, 11718, 11724, 12100, 12154, 12784, 12982, 13080, 13273, 13551, 13582, 13664, 13666, 13935, 14422, 14456, 14744, 14768, 15049, 15408, 15834, 16504, 16718, 16916, 16966, 17859, 17950, 18082, 18203, 19711, 20060, 20172, 20180, 20680, 21011, 21356, 21404, 21484, 21576, 21659, 22094, 22252, 22625, 22936, 23571, 23768, 23872, 24186, 25131, 25395, 25409, 25919, 26057, 26240, 26371, 26517, 26620, 27091, 27472, 27740, 28594, 28634, 28647, 29259, 29526]}

#glv = GroupLossVariance(X, omega, G2, 1) #axis = 1 (0 rows e 1 columns)
#RgrpIU = glv.evaluate(X_est)
#print("Group Loss Variance (Rgrp IU):", RgrpIU)