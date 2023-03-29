# VFX_hw1

## 1. Description 

藉由以不同快門速度拍攝場景，還原出 High Dynamic Range image, 描繪出 response curve 和 radiance map 表現場景的能量分佈，最後進行 tone mapping 製作出貼近電腦視覺效果的 Low Dynamic Range image.

## 2. Experiment Setup

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

<img src="https://i.imgur.com/XLaETY8.jpg" width="200px"><img src="https://i.imgur.com/1JQbvj6.jpg" width="200px"><img src="https://i.imgur.com/FnKLwuM.jpg" width="200px"><img src="https://i.imgur.com/yMrpufi.jpg" width="200px"><img src="https://i.imgur.com/VZohd2R.jpg" width="200px"><img src="https://i.imgur.com/nUZw4cB.jpg" width="200px"><img src="https://i.imgur.com/49007ct.jpg" width="200px"><img src="https://i.imgur.com/t7IC58d.jpg" width="200px"><img src="https://i.imgur.com/QMlfDZv.jpg" width="200px"><img src="https://i.imgur.com/LtenHUO.jpg" width="200px">

* 場景二：森林系館草地

<img src="https://i.imgur.com/szIOumI.jpg" width="200px"><img src="https://i.imgur.com/NRMllN5.jpg" width="200px"><img src="https://i.imgur.com/OUunUAa.jpg" width="200px"><img src="https://i.imgur.com/XgIpoYx.jpg" width="200px"><img src="https://i.imgur.com/fe7pkpk.jpg" width="200px"><img src="https://i.imgur.com/rQjkIdX.jpg" width="200px"><img src="https://i.imgur.com/IOocPoe.jpg" width="200px"><img src="https://i.imgur.com/ORKWDzd.jpg" width="200px"><img src="https://i.imgur.com/nvUwIQG.jpg" width="200px"><img src="https://i.imgur.com/ReBUKsi.jpg" width="200px">

## 3. Program Workflow

1. 讀取所有在指定路徑下的十張不同快門速度的圖片。
2. 使用 MTB 做 Image Alignment。
3. 重建 HDR 影像（Debevec or Robertson）
4. 用 Tone mapping 輸出 LDR 影像（Global, Mantiuk, Drago, Durand, or Reinhard）

## 4. Implementation Detail

### MTB Alignment
Introduction: 
Result:

### HDR Algorithm
#### (a) Debevec's method
Introduction: 



Result:


Radiance Map and Response Curve:



#### (b) Robertson's method
Introduction:


Result:


Radiance Map and Response Curve:

### Tone Mapping Algorithm
我們有嘗試五種方法做比較：Global, Mantiuk, Drago, Durand, or Reinhard

1. Global

- 場景ㄧ：森林系館走廊 gamma=0.7（左圖為 Debevec, 右圖為 Robertson）
- 場景二：森林系館草地 gamma=1.3（左圖為 Debevec, 右圖為 Robertson）

2. OpenCV: Mantiuk

- 場景ㄧ：森林系館走廊 gamma=0.7（左圖為 Debevec, 右圖為 Robertson）
- 場景二：森林系館草地 gamma=1.3（左圖為 Debevec, 右圖為 Robertson）

3. OpenCV: Drago
使用OpenCV內建tone mapping的Drago method

- 場景ㄧ：森林系館走廊 gamma=0.7（左圖為 Debevec, 右圖為 Robertson）
- 場景二：森林系館草地 gamma=1.3（左圖為 Debevec, 右圖為 Robertson）

4. OpenCV: Durand

- 場景ㄧ：森林系館走廊 gamma=0.7（左圖為 Debevec, 右圖為 Robertson）
- 場景二：森林系館草地 gamma=1.3（左圖為 Debevec, 右圖為 Robertson）

5. OpenCV: Reinhard

- 場景ㄧ：森林系館走廊 gamma=0.7（左圖為 Debevec, 右圖為 Robertson）
- 場景二：森林系館草地 gamma=1.3（左圖為 Debevec, 右圖為 Robertson）


## 6 Summary

我們完成了以下work:
- 實作 MTB algorithm
- 實作 Debevec、Robertson method
- 實作 Tone mapping 的 Global Opertor


## 7. Reproduce Steps
1. Read https://www.csie.ntu.edu.tw/~cyy/courses/vfx/23spring/assignments/proj1/
2. Know Paul Debevec's method from week 3.
3. Download sample test images from [here](http://www.mpii.mpg.de/resources/hdr/calibration/exposures.tgz) or [here](http://www.debevec.org/Research/HDR/SourceImages/Memorial_SourceImages.zip) (hdr [here](http://www.debevec.org/Research/HDR/memorial.hdr)).

4. Download our images from [here]()
5. TBD
