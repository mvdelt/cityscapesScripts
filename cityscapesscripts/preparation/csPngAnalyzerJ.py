
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


# i. ~~color.png 파일.
# colorpngpath = r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_color.png"
colorpngpath = r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000576_gtFine_color.png"

originalFormat = np.array(Image.open(colorpngpath))
print(f'j) shape originalFormat: {originalFormat.shape}')  #  (1024, 2048, 4)  <-4채널.

# i.21.3.5.17:48) ~~color.png 의 경우 4채널이라서, 쉐입 변경시켜준뒤에 np.unique 적용.
oriFormat_reshaped = originalFormat.reshape(-1,originalFormat.shape[-1])

# i.21.3.5.17:43) 넘파이의 unique 함수를 이용해서 중복되는놈들은 제거할수있음!!
segmentIds = np.unique(oriFormat_reshaped, axis=0)
print(f'j) segmentIds: {segmentIds}')
print(f'j) shape segmentIds: {segmentIds.shape}')


print('-----------------------')

# i. ~~color.png 이외의 png 파일들.
fs = [
    # r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_instanceIds.png",
    # r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_labelIds.png",
    # r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000294_gtFine_labelTrainIds.png"
    r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000576_gtFine_instanceIds.png",
    r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000576_gtFine_labelIds.png",
    r"C:\Users\starriet\Downloads\cityscapesDataset_forTestJ\gtFine\val\frankfurt\frankfurt_000000_000576_gtFine_labelTrainIds.png"
]

for f in fs:
    originalFormat = np.array(Image.open(f))
    print(f'j) shape originalFormat: {originalFormat.shape}')  # (1024, 2048)   <- 1채널이네.

    segmentIds = np.unique(originalFormat)
    print(f'j) segmentIds: {segmentIds}')
    print(f'j) shape segmentIds: {segmentIds.shape}')

    print('-------------------------')
