# Readme
## 主題
本次實作image stitiching，通過演算法找出照片的特徵點，並通過特徵點比對及stitching和blending的技術，將圖片串接形成360環繞的照片
## 實作細節
首先會使用特徵點偵測找出圖片的特徵點(Harris corner detector或MSOP)，之後進行特徵點的比對確定鄰近兩圖中重疊部分在那些位置，再來會將圖片投射至圓柱座標，經過一定程度的alignment和blending後獲得相接後的圖片
## Result
![](./figure/result.png)
## 執行
* requirement

  * 需要先有list.txt，list.txt 裡面需要有image file name與對應的focal length

  * Example : 

    ```shell
    ./data/image0.jpg 800
    ./data/image1.jpg 850
    ./data/image2.jpg 900
    .
    .
    .
    .
    ```


```shell
python3 ./src/Image_Stitching.py input [-o OUTPUT] [-m MASK] [-ord | --order| --no-order] [-f | --feature | --no-feature] [-fm | --feature_match | --no-feature_match]
```

* argv

  * input: image list mentoioned above.(ex: ./parrington/parrington/list.txt)
  * -o: output image directory.(default is ./result)
  * -m: mask size of linear blending with constant width, 0 $<$ mask size $\le$ 70.(default is -1, means barely using linear blending)
  * -ord: If images are counter-clockwise order, then use -ord.(default is False, means clockwiseorder)
  * -f: save image with features labeled on it.(default is False, and images will save to ./result/features)
  * -fm: save the features matching result between two images.(default is False, images will save to ./result/features)
  
  
* 額外套件:opencv-python/matplotlib/numpy/tqdm

ex:
* ```shell
  python3 ./src/Image_Stitching.py ./parrington/parrington/list.txt -ord
  ```
It will take some time to finished it.(5-30 min in regard of to image size)

Some stitching images are too large, so we put some demo images in link below. 
* Result
  * [stitch_origin.jpg](https://drive.google.com/file/d/1ecNLq5LF08QRznbc601VFDurU7OrPemq/view?usp=sharing)
  * [stitch_crop.jpg](https://drive.google.com/file/d/1PRwAjn21iteifR2fLN2gJdw2GiOyAjpx/view?usp=sharing)

