# Readme

* 以下資料夾路徑皆相對於hw2.py(hw2.py位於code資料夾內)
* 在hw2.py所在位置執行

```shell
python3 hw2.py [-p] [directory] [-s] [mask_size] [-o] [-1]
```

* argv

  * -p : images 所在的 directory
  * -s : linear blending with constant width 的 mask size，0 $<$ mask size $\le$ 50。default 為 -1 ，若為 -1 則使用 linear blending
  * -o : 若 image 的順序是逆時針則需要加上 "-o -1"，default 為順時針 
  
  
* 額外套件:opencv-python/matplotlib/numpy/tqdm

* requirement 

  * 需要在 images 所在的 directory 加入 list.txt，list.txt 裡面需要有 image file name 與對應的 focal length

  * Example : 

    ```shell
    image0.jpg 800
    image1.jpg 850
    image2.jpg 900
    .
    .
    .
    .
    ```

* ```shell
  python3 hw2.py -p ../data6 -s 30
  ```

  * 這行 command 會得到沒有 crop 的 result
  * 大概需要 10 分鐘才會跑出結果並存在 ../result_file/stitch_origin.jpg 

  

* ```shell
  python3 crop.py -p ../data6/stitch_origin.jpg -y 400 -h1 76 -h2 2907
  ```

  * 這行 code 會產生 crop 的 result 存在 stitch_crop.jpg
  * 由於 stitch_origin.jpg  太大，github 上沒有此 image 所以需要先 run 一次 hw2.py 的 command 或是從下面的連結下載 
* Result
  * [stitch_origin.jpg](https://drive.google.com/file/d/1ecNLq5LF08QRznbc601VFDurU7OrPemq/view?usp=sharing)
  * [stitch_crop.jpg](https://drive.google.com/file/d/1PRwAjn21iteifR2fLN2gJdw2GiOyAjpx/view?usp=sharing)

