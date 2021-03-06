
# i. 21.3.5.19:56) 
#  요약: COCO object detection 형식 -> cityscapes 의 ~~polygons.json 의 형식 으로 형식바꿔주는 코드. 
#  어떤 특정 어노테이션툴에서, 세그멘테이션 어노테이션한걸
#  COCO object detection 형식으로 내뱉었다고 했을때, 
#  그걸 cityscapes 의 ~~polygons.json 의 형식으로 바꿔주기 위한 코드임.
#  ~~polygons.json 만 있으면, cityscapesscripts 의 코드들을 사용해서 
#  원하는형식의 어노테이션png파일들 생성가능하고(cityscapes 에서 어노테이션png 파일들 종류가 몇개 있지),
#  createPanopticImgs.py 이용해서 ~~instanceIds.png 로부터
#  COCO panoptic segmentation 어노테이션형식(json & png 둘다필요)으로 변환가능함!!


# # i. COCO object detection 형식.
# {
#     # 중요만.
#     "images": [image],
#     "annotations": [annotation], 
#     "categories": [category]
# }

# image{
#     # 중요만.
#     "id": int, 
#     "width": int, ###################
#     "height": int, ###################
#     "file_name": str, 
# }

# annotation{
#     "id": int, 
#     "image_id": int, 
#     "category_id": int, 
#     "segmentation": RLE or [polygon], ###################
#     "area": float,
#     "bbox": [x,y,width,height], 
#     "iscrowd": 0 or 1,
# }

# category{
#     "id": int, 
#     "name": str, ###################
#     "supercategory": str,
# }

import json, os

# i. 불러올 어노json파일 경로. "path/to/ coco obj det annotation json file for loading"
COCO_OBJ_DET_ANNO_JSON_LOADPATH_J = r"C:\Users\starriet\Downloads\convertTestJ\panopticSegJ.json"
# i. 저장할 폴더의 경로. "path/to/ dir of cityscapes polygons.json file for saving"
CITYSCAPES_POLYGONS_JSON_SAVEDIRPATH_J = r"C:\Users\starriet\Downloads\convertTestJ"

# COCO object detection 형식의 어노json파일을 읽어들임.
with open(COCO_OBJ_DET_ANNO_JSON_LOADPATH_J) as f:
    coco_obj_det_anno = json.load(f)

# catId to catName
catId2catName = {cat["id"]: cat["name"] for cat in coco_obj_det_anno["categories"]}

# imgId to cityscapes objects
imgId2csObjects = {}
for annotation in coco_obj_det_anno["annotations"]:
    # 1 coco annotation to (maybe multiple) cityscapes polygon(s). (coco 에선 한 annotation 에 polygon 이 여러개잇을수잇음)
    for oneCocoPolygon in annotation["segmentation"]:
        csPolygon = [[x,y] for x,y in zip(oneCocoPolygon[0::2], oneCocoPolygon[1::2])]
        csObject = {
            "label": catId2catName[annotation["category_id"]],
            "polygon": csPolygon,
        }
        if annotation["image_id"] not in imgId2csObjects:
            imgId2csObjects[annotation["image_id"]] = [csObject]
        else:
            imgId2csObjects[annotation["image_id"]].append(csObject)

# 한 이미지당 하나의 (cityscapes의)~~polygons.json 파일을 만듦.
for imgDict in coco_obj_det_anno["images"]:
    cs_polygonsJson_dict = {
        "imgHeight": imgDict["width"],
        "imgWidth": imgDict["height"],
        "objects": imgId2csObjects[imgDict["id"]]
    }
    # i. make json file with cs_polygonsJson_dict.
    savePathJ = os.path.join(CITYSCAPES_POLYGONS_JSON_SAVEDIRPATH_J, \
        os.path.splitext(imgDict["file_name"])[0] + '_polygons.json')
    with open(savePathJ, 'w') as f:
        json.dump(cs_polygonsJson_dict, f)


print(f'j) finished converting.\n\
  {len(coco_obj_det_anno["images"])} images were converted from [COCO obj det format] to [cityscapes ~~polygons.json]')




# # cityscapes 의 ~~polygons.json 형식은 아래와 같음 (한 이미지당 한 json).
# {
#     "imgHeight": 1024, 
#     "imgWidth": 2048, 
#     "objects": [
#         {
#             "label": "sky", 
#             "polygon": [
#                 [
#                     704, 
#                     191
#                 ], 
#                 [
#                     1044, 
#                     404
#                 ], 
#                 [
#                     1293, 
#                     128
#                 ], 
#                 [
#                     1320, 
#                     0
#                 ], 
#                 [
#                     678, 
#                     0
#                 ], 
#                 [
#                     701, 
#                     190
#                 ]
#             ]
#         }, 
#         {
#             "label": "road", 
#             "polygon": [
#                 [
#                     1145, 
#                     391
#                 ], 
#                 [
#                     13, 
#                     467
#                 ], 
#                 [
#                     0, 
#                     466
#                 ], 
#                 [
#                     0, 
#                     1024
#                 ], 
#                 [
#                     2048, 
#                     1024
#                 ], 
#                 [
#                     2048, 
#                     446
#                 ]
#             ]
#         }, 

