


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

array = [[13,1,1,0,2,0],
         [3,9,6,0,1,0],
         [0,0,16,2,0,0],
         [0,0,0,13,0,0],
         [0,0,0,0,15,0],
         [0,0,1,0,0,15]]

df_cm = pd.DataFrame(array, index = [i for i in "ABCDEF"],columns = [i for i in "abcdef"])
# plt.figure(figsize=(10,17))
sns.set(font_scale=1.4) # for label size
# sns.heatmap(df_cm, annot=True,  cmap=sns.cm.rocket_r, annot_kws={"size": 16}) # font size
# sns.heatmap(df_cm, annot=True,  cmap="Blues", annot_kws={"size": 16}) # font size
sns.heatmap(df_cm, annot=True,  cmap="YlGnBu", annot_kws={"size": 16}) # font size
# sns.heatmap(df_cm, annot=True,  cmap="twilight", annot_kws={"size": 16}) # font size

plt.show()