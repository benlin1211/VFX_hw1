# VFX_hw1

## Description 

藉由以不同快門速度拍攝場景，還原出 High Dynamic Range image, 描繪出 response curve 和 radiance map 表現場景的能量分佈，最後進行 tone mapping 製作出貼近電腦視覺效果的 Low Dynamic Range image.

## Experiment Setup

### 相機設定


| 項目 | 描述                  |
|:---- |:--------------------- |
| 機型 | NIKON D90             |
| 鏡頭 | AF Nikkor 50mm f/1.4D |
| 焦距 | 50mm                  |
| ISO  | 200                   |
| F    | f/5                   |


### 場景圖片
| 項目   | 描述      |
|:------ |:--------- |
| 尺寸   | 2144x1424 |
| 解析度 | 300x300   |
| 數量   | 10        |

快門速度：1/4, 1/8, 1/15, 1/30, 1/60, 1/125, 1/250, 1/500, 1/1000, 1/2000

* 場景ㄧ：森林系館走廊

<img src="https://i.imgur.com/XLaETY8.jpg" width="200px">
<img src="https://i.imgur.com/1JQbvj6.jpg" width="200px">
<img src="https://i.imgur.com/FnKLwuM.jpg" width="200px">
<img src="https://i.imgur.com/yMrpufi.jpg" width="200px">
<img src="https://i.imgur.com/VZohd2R.jpg" width="200px">
<img src="https://i.imgur.com/nUZw4cB.jpg" width="200px">
<img src="https://i.imgur.com/49007ct.jpg" width="200px">
<img src="https://i.imgur.com/t7IC58d.jpg" width="200px">
<img src="https://i.imgur.com/QMlfDZv.jpg" width="200px">
<img src="https://i.imgur.com/LtenHUO.jpg" width="200px">

* 場景二：森林系館草地

<img src="https://i.imgur.com/szIOumI.jpg" width="200px"
 img src="https://i.imgur.com/NRMllN5.jpg" width="200px"
 img src="https://i.imgur.com/OUunUAa.jpg" width="200px"
 img src="https://i.imgur.com/XgIpoYx.jpg" width="200px"
 img src="https://i.imgur.com/fe7pkpk.jpg" width="200px"
 img src="https://i.imgur.com/rQjkIdX.jpg" width="200px"
 img src="https://i.imgur.com/IOocPoe.jpg" width="200px"
 img src="https://i.imgur.com/ORKWDzd.jpg" width="200px"
 img src="https://i.imgur.com/nvUwIQG.jpg" width="200px"
 img src="https://i.imgur.com/ReBUKsi.jpg" width="200px">



## Reproduce Steps
1. Read https://www.csie.ntu.edu.tw/~cyy/courses/vfx/23spring/assignments/proj1/
2. Know Paul Debevec's method from week 3.
3. Download sample test images from [here](http://www.mpii.mpg.de/resources/hdr/calibration/exposures.tgz) or [here](http://www.debevec.org/Research/HDR/SourceImages/Memorial_SourceImages.zip) (hdr [here](http://www.debevec.org/Research/HDR/memorial.hdr)).

4. Download our images from [here]()
5. TBD
