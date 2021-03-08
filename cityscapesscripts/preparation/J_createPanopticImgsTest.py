

# fileName ex: imp2_0_instanceIds.png, imp4_120_instanceIds.png

fileNames = ["imp2_0_instanceIds.png", "imp4_120_instanceIds.png"]

for fileName in fileNames:
    implDatasetGroupNumJ = fileName[3] 
    print(f'j) implDatasetGroupNumJ: {implDatasetGroupNumJ}') # "2", "4"
    implSubNumJ = fileName[len("impX_"):-len("_instanceIds.png")] 
    print(f'j) implSubNumJ: {implSubNumJ}') # "0", "120"
    
    imageId = implDatasetGroupNumJ + (3-len(implSubNumJ))*"0" + implSubNumJ
    print(f'j) imageId: {imageId}') # "2000", "4120"
    print('--------------------------')

