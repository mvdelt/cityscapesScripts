
################################################################################################################

# # fileName ex: imp2_0_instanceIds.png, imp4_120_instanceIds.png

# fileNames = ["imp2_0_instanceIds.png", "imp4_120_instanceIds.png"]

# for fileName in fileNames:
#     implDatasetGroupNumJ = fileName[3] 
#     print(f'j) implDatasetGroupNumJ: {implDatasetGroupNumJ}') # "2", "4"
#     implSubNumJ = fileName[len("impX_"):-len("_instanceIds.png")] 
#     print(f'j) implSubNumJ: {implSubNumJ}') # "0", "120"
    
#     imageId = implDatasetGroupNumJ + (3-len(implSubNumJ))*"0" + implSubNumJ
#     print(f'j) imageId: {imageId}') # "2000", "4120"
#     print('--------------------------')

################################################################################################################






################################################################################################################
# import numpy as np
# from PIL import Image

# # f = r"C:\Users\starriet\Downloads\convertTestJ\train\imp2_1_color.png"
# f = r"C:\Users\starriet\Downloads\convertTestJ\train\imp2_1_instanceIds.png"
# cs_annoPng_arrJ = np.array(Image.open(f))  # i. f 는 path/to/~~instanceIds.png
# print(f'j) cs_annoPng_arrJ.shape: {cs_annoPng_arrJ.shape}')

# segmentIds = np.unique(cs_annoPng_arrJ)
# print(f'j) segmentIds: {segmentIds}')  # i.  [   0    1    2    3    4    5    6 7000 7001 7002 7003 7004 7005 7006 8000 8001 8002 8003 9000]  # i. type: <class 'numpy.ndarray'>

################################################################################################################






################################################################################################################
import numpy as np
arr = np.array([[1,2,3],[4,5,6],[7,8,9],[5,5,5]])
# print(arr)
mask1 = arr==[4,5,6]
# print(mask1)
mask2 = arr==5
# print(mask2)

mask3 = [False, True, True, False]
print(arr[mask3])
arr[mask3]=[666,777,888] # i. 리스트를 할당해줘도 넘파이어레이로 됨.
print(arr[mask3])
print(type(arr[mask3])) # <class 'numpy.ndarray'>
print(arr)

################################################################################################################
