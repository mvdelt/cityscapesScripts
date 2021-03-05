
# i. 21.3.5.19:56) 
#  요약: COCO object detection 형식 -> cityscapes 의 ~~polygons.json 의 형식 으로 형식바꿔주는 코드. 
#  어떤 특정 어노테이션툴에서, 세그멘테이션 어노테이션한걸
#  COCO object detection 형식으로 내뱉었다고 했을때, 
#  그걸 cityscapes 의 ~~polygons.json 의 형식으로 바꿔주기 위한 코드임.
#  ~~polygons.json 만 있으면, cityscapesscripts 의 코드들을 사용해서 
#  원하는형식의 어노테이션png파일들 생성가능하고(cityscapes 에서 어노테이션png 파일들 종류가 몇개 있지),
#  createPanopticImgs.py 이용해서 ~~instanceIds.png 로부터
#  COCO panoptic segmentation 형식(json, png)으로 변환가능함!!





# cityscapes 의 ~~polygons.json 형식은 아래와 같음.
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

