

# i.21.4.26.21:06) 수플에서 위에 적어놨던 문제들(xticks 를 매트릭스의 윗부분으로 올리는거랑, 수치들 반올림문제) 모두 해결했음. SO 답변들 역시 개굿.
#  코랩에다가 코드 적어놨고, 그거 다시 여기에 붙여놓음. 
#   ->다시 이것저것 커스터마이징해서 수정. /21.4.26.23:11. 


# i.21.4.24.20:41) confusion matrix plotting test in Colab. 

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


# array = [[13.1,1,1,0,2,0.248],
#          [3.1648,9,6,0,1,0],
#          [0,0,16.14,2.1,0,0],
#          [0,0,0,13,0,0],
#          [0,0.345,0,0,15,0.002],
#          [0.011,0,1,0,0,15]]

# df_cm = pd.DataFrame(array, index = [i for i in 'ABCDEF'], columns = [i for i in 'abcdef'])




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


sn.set(rc={'figure.figsize':(7.2, 4.8)}) # i.21.4.26.22:33) 가로세로 사이즈 지정 가능.



# i. plt.gca() 는 axis 를 가져오는듯. 
#  plt.gca() 의 gca 가 아마 get current axis 인것같고, plt.gcf() 는 get current figure 인듯? 아직확인안해봄.
ax = plt.gca() 
# ax.set_xlabel('xlabelJ') # i. 안됨. 내가 지금 seaborn 쓰고있어서그런가봄. /21.4.26.22:35.

# ax.set_aspect(0.7) # aspect ratio 지정 가능한데, 난 지금 seaborn(matplotlib 위에서 작동되는듯) 쓰고있는데 얜 matplotlib 관련 함수라 그런지, 매트릭스 비율 조정은 되는데 오른쪽의 스케일바가 조정이 안됨.



# i.21.4.26.21:11) 바로여기서 labeltop=True 라고 해주면 x label (x ticks 라고도 하는것같고 정확한용어는몰겟지만) 이 위에 배치됨!!
ax.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=True, labelsize=8) # i.21.4.26.22:05) 여기서 labelsize 조절해서 label 글자크기 조절가능. 
ax.tick_params(axis="y", left=False, right=False, labelleft=True, labelright=False, labelsize=8)


# plt.figure(figsize=(10,7))
sn.set(font_scale=0.7) # for label size   # i.21.4.26.22:04) 지금내코드에선 오른쪽 스케일바 숫자 크기 조절됨. 내 x,y label 은 seaborn 이용 안하고 matplotlib 이용해서 그런것같다고 추측됨. 


# sns.heatmap(df_cm, annot=True,  cmap=sns.cm.rocket_r, annot_kws={"size": 16}) # font size
# sns.heatmap(df_cm, annot=True,  cmap="Blues", annot_kws={"size": 16}) # font size
# sns.heatmap(df_cm, annot=True,  cmap="twilight", annot_kws={"size": 16}) # font size

# i.21.4.26.21:09) 여기서 fmt 를 조정하면 자릿수,반올림 문제 해결됨!!
axJ = sn.heatmap(df_cm, annot=True, cmap='YlGnBu', fmt='g', annot_kws={"size": 8}) # font size # i.21.4.26.22:05) 매트릭스의 각 셀들의 숫자 폰트 크기조절. 
# axJ.set_title('ddddd')
axJ.tick_params(axis="x", bottom=False, top=False, labelbottom=False, labeltop=True, labelsize=8)


# axJ.set(xlabel='common xlabel', ylabel='common ylabel')
axJ.set_ylabel('ground truth', loc='center', size=11)
axJ.set_xlabel('prediction', loc='center', size=11)

axJ.xaxis.set_label_position('top')
# axJ.xaxis.set_label_coords(0,0)
axJ.xaxis.labelpad = 10
axJ.yaxis.labelpad = 8



# i. figure 를 가져옴.
#  plt.gca() 의 gca 가 아마 get current axis 인것같고, gcf 는 get current figure 인듯? 아직확인안해봄.
fig = plt.gcf()



# i. ticks 글자들 위치 미세조정.

# Create offset transform.
dx = 0/72.; dy = 0/72. # i. x,y방향 위치조정 가능./21.4.26.21:12.
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
# apply offset transform to all x ticklabels.
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

# i. same as above, but y axis.
dx = 0/72.; dy = 0/72. # i. x,y방향 위치조정 가능. /21.4.26.21:13. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
for label in ax.yaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)


# i. yticks 글자들 회전 각도.
plt.yticks(rotation=0)


# plt.savefig('/content/confMatrixJ.png', bbox_inches='tight', dpi=1000) # i. bbox_inches='tight' 로 하면 주위 여백 줄어듦. 논문용으로 dpi 1000 으로 해줌. (코랩컴에 저장하는용도로 사용했던 코드임.)
plt.savefig(r'C:\Users\starriet\1JUN\2020vision\cityscapesscripts_mvdeltGithub\cityscapesscripts\evaluation\paper2_confMatrixJ.png', bbox_inches='tight', dpi=1000)
plt.show()

