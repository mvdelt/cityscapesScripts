


# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# array = [[33,2,0,0,0,0,0,0,0,1,3], 
#         [3,31,0,0,0,0,0,0,0,0,0], 
#         [0,4,41,0,0,0,0,0,0,0,1], 
#         [0,1,0,30,0,6,0,0,0,0,1], 
#         [0,0,0,0,38,10,0,0,0,0,0], 
#         [0,0,0,3,1,39,0,0,0,0,4], 
#         [0,2,2,0,4,1,31,0,0,0,2],
#         [0,1,0,0,0,0,0,36,0,2,0], 
#         [0,0,0,0,0,0,1,5,37,5,1], 
#         [3,0,0,0,0,0,0,0,0,39,0], 
#         [0,0,0,0,0,0,0,0,0,0,38]]
# df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJK"],
#                   columns = [i for i in "ABCDEFGHIJK"])
# plt.figure(figsize = (10,7))
# sns.heatmap(df_cm, annot=True)

# plt.show()




import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# array = [[13,1,1,0,2,0],
#          [3,9,6,0,1,0],
#          [0,0,16,2,0,0],
#          [0,0,0,13,0,0],
#          [0,0,0,0,15,0],
#          [0,0,1,0,0,15]]

array = [[0.981, 0.006, 0.002, 0.006, 0.   , 0.002, 0.002, 0.001, 0.537],
         [0.037, 0.941, 0.002, 0.   , 0.016, 0.002, 0.001, 0.001, 0.222],
         [0.018, 0.012, 0.902, 0.015, 0.   , 0.018, 0.03 , 0.005, 0.046],
         [0.023, 0.   , 0.017, 0.959, 0.   , 0.   , 0.   , 0.001, 0.071],
         [0.001, 0.194, 0.   , 0.   , 0.802, 0.   , 0.   , 0.002, 0.017],
         [0.018, 0.025, 0.017, 0.   , 0.   , 0.897, 0.043, 0.   , 0.039],
         [0.019, 0.041, 0.011, 0.   , 0.   , 0.054, 0.867, 0.008, 0.038],
         [0.029, 0.004, 0.003, 0.   , 0.   , 0.001, 0.038, 0.924, 0.03 ]]

df_cm = pd.DataFrame(array, \
    index = ['UL', 'Man', 'Max', 'Sinus', 'Canal', 'T_n', 'T_tx', 'Impl'], \
    columns = ['UL', 'Man', 'Max', 'Sinus', 'Canal', 'T_n', 'T_tx', 'Impl', 'Prior'])
# plt.figure(figsize=(10,17))
sns.set(font_scale=0.8) # for label size
# sns.heatmap(df_cm, annot=True,  cmap=sns.cm.rocket_r, annot_kws={"size": 16}) # font size
# sns.heatmap(df_cm, annot=True,  cmap="Blues", annot_kws={"size": 16}) # font size
sns.heatmap(df_cm, annot=True,  cmap="YlGnBu", annot_kws={"size": 10}) # font size
# sns.heatmap(df_cm, annot=True,  cmap="twilight", annot_kws={"size": 16}) # font size


# i.21.4.24.23:45) 코랩에서 돌리려면 또 깃헙 업뎃해서 다시 풀 하고 그래야해서, 걍 여기서 플롯 그려보는중.
#  plot 에서, 아직 못한거: 
#  1. xticks (x축 라벨) 를 위로 올리고싶은데, 어케함??
#  2. 수치들이 왜 높은값들은 반올림돼잇음??


plt.yticks(rotation=0) 
plt.xticks(rotation=0) 

plt.show()