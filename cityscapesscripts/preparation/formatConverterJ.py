
# i. 21.3.5.19:56) 
#  요약: COCO object detection 형식 -> cityscapes 의 ~~polygons.json 의 형식 으로 형식바꿔주는 코드. 
#  어떤 특정 어노테이션툴에서, 세그멘테이션 어노테이션한걸
#  COCO object detection 형식으로 내뱉었다고 했을때, 
#  그걸 cityscapes 의 ~~polygons.json 의 형식으로 바꿔주기 위한 코드임.
#  ~~polygons.json 만 있으면, cityscapesscripts 의 코드들을 사용해서 
#  원하는형식의 어노테이션png파일들 생성가능하고(cityscapes 에서 어노테이션png 파일들 종류가 몇개 있지),
#  createPanopticImgs.py 이용해서 ~~instanceIds.png 로부터
#  COCO panoptic segmentation 어노테이션형식(json & png 둘다필요)으로 변환가능함!!


# i. COCO object detection 형식.
{
    # 중요만.
    "images": [image],
    "annotations": [annotation], 
    "categories": [category]
}

image{
    # 중요만.
    "id": int, 
    "width": int, ###################
    "height": int, ###################
    "file_name": str, 
}

annotation{
    "id": int, 
    "image_id": int, 
    "category_id": int, 
    "segmentation": RLE or [polygon], ###################
    "area": float,
    "bbox": [x,y,width,height], 
    "iscrowd": 0 or 1,
}

category{
    "id": int, 
    "name": str, ###################
    "supercategory": str,
}

# coco polygon to cityscapes polygon 
#  -> 근데 coco 에선 한 annotations 에 polygon 이 여러개잇을수잇는데..

# catId to catName
catId2catName = {cat["id"]: cat["name"] for cat in coco_obj_det_anno["categories"]}

# imgId to objects
imgId2objects = {}
for annotation in coco_obj_det_anno["annotation"]:
    obj = {
        "label": catId2catName[annotation["category_id"]],
        "polygon": ,
    }
    imgId2objects annotation["image_id"]


# 위의 COCO object detection json파일을 deserialize 한것을 coco_obj_det_anno 라고하면,
for imgDict in coco_obj_det_anno["images"]:
    polygonsDict = {
        "imgHeight": imgDict["width"],
        "imgWidth": imgDict["height"],
        "objects":
    }

for annotation in coco_obj_det_anno["annotations"]:
    annotation["image_id"]






# cityscapes 의 ~~polygons.json 형식은 아래와 같음 (한 이미지당 한 json).
{
    "imgHeight": 1024, 
    "imgWidth": 2048, 
    "objects": [
        {
            "label": "sky", 
            "polygon": [
                [
                    704, 
                    191
                ], 
                [
                    1044, 
                    404
                ], 
                [
                    1293, 
                    128
                ], 
                [
                    1320, 
                    0
                ], 
                [
                    678, 
                    0
                ], 
                [
                    701, 
                    190
                ]
            ]
        }, 
        {
            "label": "road", 
            "polygon": [
                [
                    1145, 
                    391
                ], 
                [
                    13, 
                    467
                ], 
                [
                    0, 
                    466
                ], 
                [
                    0, 
                    1024
                ], 
                [
                    2048, 
                    1024
                ], 
                [
                    2048, 
                    446
                ]
            ]
        }, 

