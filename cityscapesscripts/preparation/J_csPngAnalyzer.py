
# i.21.3.5.17:44) cityscapes 데이터셋에서 gtFine폴더안에 png파일 3개에다가,
#  createTrainIdLabelImgs.py 사용하면 생성되는 labelTrainIds.png 까지 총 png파일 4개에 대해서 
#  넘파이의 unique 함수를 이용해서 각 png파일에 담겨진 클래스들(person, car, etc.)의 갯수를 살펴본 코드. //21.3.4.쯤에 첨 작성한 코드임.


# python imports
from __future__ import print_function, absolute_import, division, unicode_literals
import os
import glob
import sys
import argparse
import json
import numpy as np

# Image processing
from PIL import Image




###########################################################################################################################################################

# # i. ~~color.png 파일.
# # colorpngpath = r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_color.png"
# colorpngpath = r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000576_gtFine_color.png"

# originalFormat = np.array(Image.open(colorpngpath))
# print(f'j) shape originalFormat: {originalFormat.shape}')  #  (1024, 2048, 4)  <-4채널.

# # i.21.3.5.17:48) ~~color.png 의 경우 4채널이라서, 쉐입 변경시켜준뒤에 np.unique 적용.
# oriFormat_reshaped = originalFormat.reshape(-1,originalFormat.shape[-1])

# # i.21.3.5.17:43) 넘파이의 unique 함수를 이용해서 중복되는놈들은 제거할수있음!!
# segmentIds = np.unique(oriFormat_reshaped, axis=0)
# print(f'j) segmentIds: {segmentIds}')
# print(f'j) shape segmentIds: {segmentIds.shape}')


# print('-----------------------')

# # i. ~~color.png 이외의 png 파일들.
# fs = [
#     # r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_instanceIds.png",
#     # r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_labelIds.png",
#     # r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_labelTrainIds.png"
#     r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000576_gtFine_instanceIds.png",
#     r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000576_gtFine_labelIds.png",
#     r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000576_gtFine_labelTrainIds.png"
# ]

# for f in fs:
#     originalFormat = np.array(Image.open(f))
#     print(f'j) shape originalFormat: {originalFormat.shape}')  # (1024, 2048)   <- 1채널이네.

#     segmentIds = np.unique(originalFormat)
#     print(f'j) segmentIds: {segmentIds}')
#     print(f'j) shape segmentIds: {segmentIds.shape}')

#     print('-------------------------')

###########################################################################################################################################################




###########################################################################################################################################################
# # i.21.3.9.11:00) J_createPanopticImgs.py 실행결과 체크해보는중. ->이상없음.

# i.21.3.11.12:43) 기존의 convertTestJ 에서 panopticSeg_dentPanoJ 로 폴더명 바꿨고, 그안에 gt 및 inputOriPano 두가지 폴더 다시 만들어줬음.
#  따라서, 지금 요 경로들은 이제 적용안됨. 바꿔줘야함.
# f = r"C:\Users\starriet\Downloads\convertTestJ\J_cocoformat_panoptic_train\imp2_1_panopticAnno.png"
# originalFormat = np.array(Image.open(f))
# print(f'j) originalFormat.shape: {originalFormat.shape}')  # (976, 1976, 3)

# # i. 다채널이라서, 쉐입 변경시켜준뒤에 np.unique 적용.
# oriFormat_reshaped = originalFormat.reshape((-1, originalFormat.shape[-1]))
# print(f'j) oriFormat_reshaped.shape: {oriFormat_reshaped.shape}')

# segmentIds = np.unique(oriFormat_reshaped, axis=0)
# print(f'j) segmentIds: {segmentIds}')
# print(f'j) segmentIds.shape: {segmentIds.shape}')

# for idRGB in segmentIds:
#     id = idRGB[0] + idRGB[1]*256 + idRGB[2]*(256^2)
#     print(f'j) id in number: {id}')

# print('-------------------------')

###########################################################################################################################################################




###########################################################################################################################################################
# i.21.3.10.21:28) ~~labelTrainIds.png 만들어준거 제대로 된건지 체크. (~~labelTrainIds.png 도 만들어줘야한다는걸 알게돼서, 만들어줬음.)
#  -> 제대로만들어졌네.
# i.21.3.11.12:43) 기존의 convertTestJ 에서 panopticSeg_dentPanoJ 로 폴더명 바꿨고, 그안에 gt 및 inputOriPano 두가지 폴더 다시 만들어줬음.
#  따라서, 지금 요 경로들은 이제 적용안됨. 바꿔줘야함.
f = r"C:\Users\starriet\Downloads\convertTestJ\train\imp2_1_labelTrainIds.png"
originalFormat = np.array(Image.open(f))
print(f'j) originalFormat.shape: {originalFormat.shape}') # (976, 1976)

segmentIds = np.unique(originalFormat)
print(f'j) segmentIds: {segmentIds}') # [0 1 2 3 4 5 6 7 8 9]  # i. <- 내가 만들어준 10개의 카테고리 (labels.py 에도 작성해둔) 잘 기록되어있는것을 알수있음.
print(f'j) segmentIds.shape: {segmentIds.shape}') # (10,)

print('-------------------------')

###########################################################################################################################################################
