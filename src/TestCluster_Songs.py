
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import pair_confusion_matrix


#G_Movies_1M_All = {1: [123, 148, 216, 245, 329, 550, 660, 731, 889, 984, 1015, 1019, 1181, 1207, 1224, 1274, 1285, 1298, 1317, 1422, 1449, 1451, 1465, 1671, 1680, 1698, 1737, 1741, 1748, 1780, 1880, 1884, 1941, 1980, 2092, 2304, 2544, 2665, 2793, 2962, 3018, 3272, 3336, 3401, 3410, 3476, 3526, 3618, 3675, 3705, 3724, 3823, 3841, 3999, 4016, 4021, 4054, 4064, 4083, 4169, 4238, 4277, 4354, 4425, 4447, 4510, 4647, 4725, 4802, 4808, 4979, 5015, 5054, 5100, 5111, 5493, 5504, 5511, 5550, 5675, 5916, 5954, 6016], 2: [319, 411, 528, 531, 692, 721, 752, 855, 1051, 1068, 1088, 1125, 1150, 1194, 1203, 1242, 1354, 1383, 1448, 1605, 1675, 1812, 1897, 1899, 1920, 1926, 1958, 1988, 2073, 2106, 2529, 2820, 2878, 2907, 2934, 3067, 3224, 3292, 3308, 3312, 3462, 3483, 3519, 3539, 3562, 3589, 3829, 3842, 4048, 4085, 4305, 4345, 4387, 4482, 4579, 5107, 5306, 5333, 5387, 5433, 5530, 5605, 5643, 5759, 5763, 5812, 5878, 5996, 6036], 3: [48, 53, 149, 173, 195, 202, 302, 308, 331, 352, 424, 438, 482, 509, 524, 533, 543, 549, 678, 699, 710, 770, 839, 869, 877, 881, 1010, 1050, 1069, 1112, 1117, 1120, 1137, 1246, 1264, 1266, 1303, 1333, 1340, 1425, 1447, 1470, 1496, 1579, 1613, 1632, 1635, 1647, 1676, 1733, 1749, 1764, 1835, 1837, 1889, 1912, 1943, 2010, 2012, 2015, 2030, 2063, 2077, 2109, 2116, 2124, 2181, 2453, 2507, 2777, 2857, 2887, 2909, 2986, 3029, 3032, 3118, 3163, 3182, 3280, 3285, 3311, 3320, 3389, 3391, 3471, 3475, 3491, 3507, 3610, 3626, 3648, 3650, 3681, 3683, 3693, 3768, 3778, 3792, 3808, 3821, 3824, 3834, 3884, 3885, 3929, 3934, 3942, 4033, 4041, 4089, 4140, 4186, 4227, 4312, 4344, 4386, 4411, 4448, 4508, 4543, 4578, 4682, 4728, 4732, 4867, 4957, 5011, 5026, 5046, 5074, 5220, 5256, 5312, 5367, 5394, 5501, 5536, 5614, 5627, 5636, 5682, 5747, 5788, 5795, 5831, 5880, 5888]}
G_Movies_1M_AllAndLoss = {1: [319, 411, 528, 531, 692, 721, 752, 770, 855, 1019, 1051, 1068, 1069, 1088, 1125, 1150, 1194, 1203, 1242, 1340, 1354, 1383, 1425, 1448, 1470, 1605, 1675, 1812, 1897, 1899, 1920, 1926, 1958, 1988, 2015, 2073, 2106, 2529, 2665, 2820, 2878, 2907, 2934, 3067, 3224, 3292, 3308, 3312, 3462, 3476, 3483, 3519, 3526, 3539, 3562, 3589, 3610, 3829, 3842, 4048, 4085, 4305, 4345, 4354, 4387, 4482, 4579, 4647, 4682, 4979, 5074, 5107, 5111, 5306, 5333, 5387, 5433, 5530, 5605, 5643, 5759, 5763, 5812, 5878, 5954, 5996, 6036], 2: [148, 216, 550, 660, 731, 889, 984, 1015, 1181, 1207, 1224, 1274, 1285, 1317, 1449, 1465, 1680, 1748, 1884, 1941, 1980, 2092, 2304, 2544, 3618, 3705, 3823, 3841, 4016, 4021, 4064, 4083, 4169, 4277, 4447, 4510, 4802, 5100, 5504, 5511, 5916, 6016], 3: [48, 53, 123, 149, 173, 195, 202, 245, 302, 308, 329, 331, 352, 424, 438, 482, 509, 524, 533, 543, 549, 678, 699, 710, 839, 869, 877, 881, 1010, 1050, 1112, 1117, 1120, 1137, 1246, 1264, 1266, 1298, 1303, 1333, 1422, 1447, 1451, 1496, 1579, 1613, 1632, 1635, 1647, 1671, 1676, 1698, 1733, 1737, 1741, 1749, 1764, 1780, 1835, 1837, 1880, 1889, 1912, 1943, 2010, 2012, 2030, 2063, 2077, 2109, 2116, 2124, 2181, 2453, 2507, 2777, 2793, 2857, 2887, 2909, 2962, 2986, 3018, 3029, 3032, 3118, 3163, 3182, 3272, 3280, 3285, 3311, 3320, 3336, 3389, 3391, 3401, 3410, 3471, 3475, 3491, 3507, 3626, 3648, 3650, 3675, 3681, 3683, 3693, 3724, 3768, 3778, 3792, 3808, 3821, 3824, 3834, 3884, 3885, 3929, 3934, 3942, 3999, 4033, 4041, 4054, 4089, 4140, 4186, 4227, 4238, 4312, 4344, 4386, 4411, 4425, 4448, 4508, 4543, 4578, 4725, 4728, 4732, 4808, 4867, 4957, 5011, 5015, 5026, 5046, 5054, 5220, 5256, 5312, 5367, 5394, 5493, 5501, 5536, 5550, 5614, 5627, 5636, 5675, 5682, 5747, 5788, 5795, 5831, 5880, 5888]}
G_Movies_1M_Loss = {1: [752, 770, 1051, 1069, 1242, 1340, 1470, 1812, 2015, 2073, 2665, 2907, 3182, 3308, 3312, 3475, 3476, 3526, 3589, 3610, 3842, 4048, 4083, 4354, 4387, 4979, 5074, 5605, 5763, 5795, 5878, 5954], 2: [148, 202, 216, 302, 329, 352, 411, 438, 482, 524, 528, 531, 533, 543, 549, 550, 699, 721, 731, 869, 881, 889, 1010, 1019, 1068, 1125, 1137, 1150, 1207, 1224, 1246, 1266, 1303, 1333, 1354, 1383, 1425, 1447, 1448, 1449, 1632, 1671, 1675, 1680, 1741, 1749, 1764, 1912, 1920, 1926, 1941, 1943, 2010, 2063, 2077, 2092, 2106, 2109, 2116, 2124, 2181, 2453, 2507, 2529, 2777, 2793, 2820, 2878, 2887, 2909, 3032, 3067, 3280, 3285, 3311, 3320, 3391, 3401, 3410, 3462, 3491, 3507, 3519, 3562, 3648, 3650, 3681, 3683, 3792, 3808, 3823, 3829, 3834, 3841, 3884, 3885, 3929, 3942, 3999, 4016, 4021, 4041, 4064, 4085, 4140, 4186, 4227, 4305, 4344, 4386, 4411, 4508, 4510, 4578, 4647, 4682, 5015, 5026, 5046, 5107, 5111, 5306, 5312, 5333, 5367, 5387, 5433, 5493, 5501, 5504, 5530, 5550, 5614, 5643, 5682, 5759, 5788, 5812, 5880, 5888, 5996], 3: [48, 53, 123, 149, 173, 195, 245, 308, 319, 331, 424, 509, 660, 678, 692, 710, 839, 855, 877, 984, 1015, 1050, 1088, 1112, 1117, 1120, 1181, 1194, 1203, 1264, 1274, 1285, 1298, 1317, 1422, 1451, 1465, 1496, 1579, 1605, 1613, 1635, 1647, 1676, 1698, 1733, 1737, 1748, 1780, 1835, 1837, 1880, 1884, 1889, 1897, 1899, 1958, 1980, 1988, 2012, 2030, 2304, 2544, 2857, 2934, 2962, 2986, 3018, 3029, 3118, 3163, 3224, 3272, 3292, 3336, 3389, 3471, 3483, 3539, 3618, 3626, 3675, 3693, 3705, 3724, 3768, 3778, 3821, 3824, 3934, 4033, 4054, 4089, 4169, 4238, 4277, 4312, 4345, 4425, 4447, 4448, 4482, 4543, 4579, 4725, 4728, 4732, 4802, 4808, 4867, 4957, 5011, 5054, 5100, 5220, 5256, 5394, 5511, 5536, 5627, 5636, 5675, 5747, 5831, 5916, 6016, 6036]}
G_Movies_1M_NR = {1: [48, 53, 149, 173, 195, 202, 302, 308, 331, 352, 424, 438, 482, 509, 524, 528, 531, 533, 543, 549, 678, 692, 699, 710, 752, 770, 839, 855, 869, 877, 881, 1010, 1050, 1051, 1068, 1069, 1088, 1112, 1117, 1120, 1125, 1137, 1150, 1194, 1203, 1246, 1264, 1266, 1303, 1333, 1340, 1354, 1383, 1425, 1447, 1448, 1470, 1496, 1579, 1605, 1613, 1632, 1635, 1647, 1675, 1676, 1680, 1733, 1749, 1764, 1812, 1835, 1837, 1889, 1912, 1926, 1943, 1958, 1988, 2010, 2012, 2015, 2030, 2063, 2073, 2077, 2106, 2109, 2116, 2124, 2181, 2453, 2507, 2777, 2857, 2887, 2909, 2986, 3029, 3032, 3067, 3118, 3163, 3182, 3224, 3280, 3285, 3308, 3311, 3312, 3320, 3389, 3391, 3462, 3471, 3475, 3491, 3507, 3519, 3539, 3562, 3610, 3626, 3648, 3650, 3681, 3683, 3693, 3768, 3778, 3792, 3808, 3821, 3824, 3829, 3834, 3884, 3885, 3929, 3934, 3942, 4033, 4041, 4085, 4089, 4140, 4186, 4227, 4312, 4344, 4345, 4386, 4387, 4411, 4448, 4508, 4543, 4578, 4579, 4682, 4728, 4732, 4867, 4957, 5011, 5026, 5046, 5074, 5220, 5256, 5306, 5312, 5333, 5367, 5387, 5394, 5501, 5530, 5536, 5605, 5614, 5627, 5636, 5682, 5747, 5759, 5763, 5788, 5795, 5812, 5831, 5878, 5880, 5888, 5996, 6036], 2: [148, 216, 319, 411, 550, 660, 731, 889, 984, 1207, 1224, 1274, 1317, 1465, 1748, 1884, 1897, 1899, 2092, 2304, 2544, 2878, 3292, 3476, 3483, 3589, 3618, 3705, 3823, 3841, 4016, 4021, 4064, 4083, 4169, 4305, 4447, 4510, 4802, 5100, 5107, 5504, 5511, 5916, 5954, 6016], 3: [123, 245, 329, 721, 1015, 1019, 1181, 1242, 1285, 1298, 1422, 1449, 1451, 1671, 1698, 1737, 1741, 1780, 1880, 1920, 1941, 1980, 2529, 2665, 2793, 2820, 2907, 2934, 2962, 3018, 3272, 3336, 3401, 3410, 3526, 3675, 3724, 3842, 3999, 4048, 4054, 4238, 4277, 4354, 4425, 4482, 4647, 4725, 4808, 4979, 5015, 5054, 5111, 5433, 5493, 5550, 5643, 5675]}

Labels_Movies_1M_Age          = [0, 0, 2, 1, 0, 0, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 1, 2, 0, 0, 0, 1, 2, 2, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 2, 1, 0, 0, 2, 0, 0, 0, 2, 1, 0, 1, 1, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 1, 2, 0, 2, 2, 0, 1, 0, 2, 0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 2, 0, 0, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 2, 2, 1, 0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 2, 0, 1, 0, 0, 0, 2, 0, 2, 0, 0, 1, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]
Labels_Movies_1M_AgeAndGender = [2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 0, 1, 2, 2, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 1, 1, 2, 0, 2, 2, 0, 1, 0, 2, 2, 0, 2, 2, 2, 1, 1, 2, 1, 1, 2, 0, 0, 2, 0, 2, 2, 2, 0, 2, 0, 1, 0, 0, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 0, 0, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2, 0, 2, 2, 2, 2, 1, 0, 2, 2, 1, 2, 1, 1, 1, 2, 2, 1, 0, 2, 2, 1, 1, 2, 0, 0, 2, 0, 0, 1, 2, 0, 1, 0, 2, 2, 2, 2, 2, 0, 2, 1, 0, 2, 2, 2, 2, 1, 2, 2, 0, 1, 1, 2, 1, 0, 2, 0, 2, 0, 2, 0, 1, 2, 1, 2, 2, 0, 2, 2, 2, 0, 1, 2, 2, 0, 0, 2, 0, 2, 1, 2, 2, 1, 1, 0, 2, 2, 1, 0, 2, 2, 0, 1, 0, 0, 0, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 2, 1, 0, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0, 2, 2, 1, 2, 2, 1, 1, 0, 2, 2, 0, 1, 2, 0, 2, 1, 1, 2, 0, 2, 1, 2, 2, 0, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 0, 1, 2, 2, 0, 2, 0, 2, 0, 2, 0, 1, 2, 1, 1, 0, 2, 1, 0, 2, 2, 2, 0, 1, 2, 2, 0, 0, 2, 2, 0, 2, 0, 2, 2, 1, 1, 0, 1, 0]
Labels_Movies_1M_All          = [2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 1, 0, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 0, 0, 2, 1, 2, 2, 1, 0, 1, 2, 2, 1, 2, 2, 2, 0, 0, 2, 0, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 0, 1, 1, 0, 0, 1, 2, 2, 2, 0, 0, 0, 2, 0, 2, 2, 1, 1, 0, 2, 2, 1, 0, 0, 0, 2, 2, 2, 1, 2, 2, 2, 2, 0, 1, 2, 0, 0, 2, 0, 0, 0, 2, 2, 0, 1, 2, 2, 0, 0, 2, 1, 1, 2, 1, 1, 0, 2, 1, 0, 1, 2, 2, 2, 2, 2, 1, 2, 0, 1, 2, 2, 2, 2, 0, 2, 2, 1, 0, 0, 2, 0, 1, 2, 1, 2, 1, 2, 1, 0, 2, 0, 2, 2, 1, 2, 2, 2, 1, 0, 2, 2, 1, 1, 2, 1, 2, 0, 2, 2, 0, 0, 1, 2, 2, 0, 1, 2, 2, 1, 0, 1, 1, 1, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 1, 2, 0, 1, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 1, 0, 0, 0, 1, 2, 2, 0, 2, 2, 0, 0, 1, 2, 2, 1, 0, 2, 1, 2, 0, 0, 2, 1, 2, 0, 2, 2, 1, 0, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 0, 2, 2, 0, 2, 0, 1, 0, 2, 2, 1, 2, 1, 2, 1, 2, 1, 0, 2, 0, 0, 1, 2, 0, 1, 2, 2, 2, 1, 0, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 0, 0, 1, 0, 1]
Labels_Movies_1M_AllAndLoss   = [2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 1, 1, 2, 0, 2, 2, 0, 1, 0, 0, 2, 0, 2, 2, 2, 1, 1, 2, 1, 0, 2, 0, 0, 0, 0, 2, 2, 2, 0, 2, 0, 1, 0, 0, 1, 1, 0, 2, 2, 2, 1, 1, 2, 2, 1, 2, 0, 0, 0, 2, 0, 2, 0, 1, 2, 1, 0, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 0, 0, 2, 0, 0, 1, 2, 0, 1, 0, 2, 2, 0, 2, 2, 0, 2, 1, 0, 2, 2, 2, 2, 1, 2, 2, 0, 1, 0, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 0, 2, 1, 0, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 0, 2, 1, 1, 0, 2, 2, 1, 2, 2, 2, 1, 0, 2, 2, 0, 0, 2, 0, 2, 2, 1, 2, 0, 2, 1, 2, 2, 0, 0, 0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 1, 0, 0, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 1, 1, 0, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 0, 2, 0, 2, 2, 1, 0, 0, 1, 0]
Labels_Movies_1M_Gender       = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0]
Labels_Movies_1M_Loss         = [2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 1, 1, 0, 0, 2, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 0, 1, 0, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 0, 1, 2, 1, 2, 2, 2, 1, 2, 1, 0, 1, 1, 2, 1, 1, 1, 1, 2, 2, 0, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 0, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 2, 1, 1, 0, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 0, 2, 2, 1, 1, 2, 0, 1, 0, 1, 2, 2, 1, 1, 1, 1, 2, 0, 0, 2, 1, 1, 1, 0, 2, 1, 0, 0, 2, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 0, 2, 1, 0, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 0, 1, 0, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 1, 1, 1, 2, 0, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1, 0, 1, 2, 2, 1, 2, 1, 2, 1, 0, 1, 0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 2]
Labels_Movies_1M_NR           = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 2, 1, 2, 1, 2, 1, 1, 0, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 0, 1, 2, 0, 2, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 0, 1, 2, 0, 1, 1, 1, 2, 1, 0, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 0, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 0, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 2, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 0, 1, 0, 2, 0, 1, 2, 0, 1, 1, 2, 1, 1, 2, 2, 2, 1, 0, 0, 2, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 0, 2, 0, 1, 1, 1, 1, 2, 1, 2, 2]

'''
sklearn.metrics.cluster.adjusted_rand_score: Rand index
Given the knowledge of the ground truth class assignments labels_true and our clustering algorithm assignments of the same 
samples labels_pred, the (adjusted or unadjusted) Rand index is a function that measures the similarity of the two assignments, 
ignoring permutations
'''
print('--------------------------------------------------------------------------')
print('sklearn.metrics.cluster: adjusted_rand_score')
print('--------------------------------------------------------------------------')
print('Labels_Movies_1M_Age vs Labels_Movies_1M_Loss:          ', adjusted_rand_score(Labels_Movies_1M_Age, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_AgeAndGender vs Labels_Movies_1M_Loss: ', adjusted_rand_score(Labels_Movies_1M_AgeAndGender, Labels_Movies_1M_Loss))
print('Label_Movies_1M_All vs Labels_Movies_1M_Loss:           ', adjusted_rand_score(Labels_Movies_1M_All, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_AllAndLoss vs Labels_Movies_1M_Loss:   ', adjusted_rand_score(Labels_Movies_1M_AllAndLoss, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_Gender vs Labels_Movies_1M_Loss:       ', adjusted_rand_score(Labels_Movies_1M_Gender, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_NR vs Labels_Movies_1M_Loss:           ', adjusted_rand_score(Labels_Movies_1M_NR, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_Loss vs Labels_Movies_1M_Loss:         ', adjusted_rand_score(Labels_Movies_1M_Loss, Labels_Movies_1M_Loss))
print('--------------------------------------------------------------------------')

print('--------------------------------------------------------------------------')
print('sklearn.metrics.cluster: rand_score')
print('--------------------------------------------------------------------------')
print('Labels_Movies_1M_Age vs Labels_Movies_1M_Loss:          ', rand_score(Labels_Movies_1M_Age, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_AgeAndGender vs Labels_Movies_1M_Loss: ', rand_score(Labels_Movies_1M_AgeAndGender, Labels_Movies_1M_Loss))
print('Label_Movies_1M_All vs Labels_Movies_1M_Loss:           ', rand_score(Labels_Movies_1M_All, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_AllAndLoss vs Labels_Movies_1M_Loss:   ', rand_score(Labels_Movies_1M_AllAndLoss, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_Gender vs Labels_Movies_1M_Loss:       ', rand_score(Labels_Movies_1M_Gender, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_NR vs Labels_Movies_1M_Loss:           ', rand_score(Labels_Movies_1M_NR, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_Loss vs Labels_Movies_1M_Loss:         ', rand_score(Labels_Movies_1M_Loss, Labels_Movies_1M_Loss))
print('--------------------------------------------------------------------------')

'''
sklearn.metrics.cluster.adjusted_mutual_info_score: Mutual Information based scores
Given the knowledge of the ground truth class assignments labels_true and our clustering algorithm assignments of the same 
samples labels_pred, the Mutual Information is a function that measures the agreement of the two assignments, ignoring permutations.
'''
print()
print('--------------------------------------------------------------------------')
print('sklearn.metrics.cluster: adjusted_mutual_info_score')
print('--------------------------------------------------------------------------')
print('Labels_Movies_1M_Age vs Labels_Movies_1M_Loss:          ', adjusted_mutual_info_score(Labels_Movies_1M_Age, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_AgeAndGender vs Labels_Movies_1M_Loss: ', adjusted_mutual_info_score(Labels_Movies_1M_AgeAndGender, Labels_Movies_1M_Loss))
print('Label_Movies_1M_All vs Labels_Movies_1M_Loss:           ', adjusted_mutual_info_score(Labels_Movies_1M_All, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_AllAndLoss vs Labels_Movies_1M_Loss:   ', adjusted_mutual_info_score(Labels_Movies_1M_AllAndLoss, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_Gender vs Labels_Movies_1M_Loss:       ', adjusted_mutual_info_score(Labels_Movies_1M_Gender, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_NR vs Labels_Movies_1M_Loss:           ', adjusted_mutual_info_score(Labels_Movies_1M_NR, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_Loss vs Labels_Movies_1M_Loss:         ', adjusted_mutual_info_score(Labels_Movies_1M_Loss, Labels_Movies_1M_Loss))
print('--------------------------------------------------------------------------')

'''
sklearn.metrics.cluster.fowlkes_mallows_score: Fowlkes-Mallows scores
The Fowlkes-Mallows index (sklearn.metrics.fowlkes_mallows_score) can be used when the ground truth class assignments of the 
samples is known. The Fowlkes-Mallows score FMI is defined as the geometric mean of the pairwise precision and recall.
'''
print()
print('--------------------------------------------------------------------------')
print('sklearn.metrics.cluster: fowlkes_mallows_score')
print('--------------------------------------------------------------------------')
print('Labels_Movies_1M_Age vs Labels_Movies_1M_Loss:          ', fowlkes_mallows_score(Labels_Movies_1M_Age, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_AgeAndGender vs Labels_Movies_1M_Loss: ', fowlkes_mallows_score(Labels_Movies_1M_AgeAndGender, Labels_Movies_1M_Loss))
print('Label_Movies_1M_All vs Labels_Movies_1M_Loss:           ', fowlkes_mallows_score(Labels_Movies_1M_All, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_AllAndLoss vs Labels_Movies_1M_Loss:   ', fowlkes_mallows_score(Labels_Movies_1M_AllAndLoss, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_Gender vs Labels_Movies_1M_Loss:       ', fowlkes_mallows_score(Labels_Movies_1M_Gender, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_NR vs Labels_Movies_1M_Loss:           ', fowlkes_mallows_score(Labels_Movies_1M_NR, Labels_Movies_1M_Loss))
print('Labels_Movies_1M_Loss vs Labels_Movies_1M_Loss:         ', fowlkes_mallows_score(Labels_Movies_1M_Loss, Labels_Movies_1M_Loss))
print('--------------------------------------------------------------------------')

'''
sklearn.metrics.cluster.contingency_matrix: Contingency Matrix
Contingency matrix (sklearn.metrics.cluster.contingency_matrix) reports the intersection cardinality for every 
true/predicted cluster pair.
'''
print()
print('--------------------------------------------------------------------------')
print('contingency_matrix')
cont_matrix = contingency_matrix(Labels_Movies_1M_Age, Labels_Movies_1M_Loss)
print(cont_matrix)
print('--------------------------------------------------------------------------')

'''
sklearn.metrics.cluster.pair_confusion_matrix: Pair Confusion Matrix
The pair confusion matrix is a 2x2
between two clusterings computed by considering all pairs of samples and counting pairs that are assigned 
into the same or into different clusters under the true and predicted clusterings.
'''
print()
print('--------------------------------------------------------------------------')
print('pair_confusion_matrix')
print('--------------------------------------------------------------------------')
print(pair_confusion_matrix(Labels_Movies_1M_Age, Labels_Movies_1M_Loss))
print('--------------------------------------------------------------------------')