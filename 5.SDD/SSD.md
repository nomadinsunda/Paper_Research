# SSD : Single Shot MultiBox Detector

## 1. Introduction

- 1-stage detector인 YOLO는 45 frames per second(FPS:Frame Per Sec)로 7 FPS인 Faster R-CNN보다 **속도가 크게 향상**되었지만 <br>
YOLO mAP : 63.4%과 Faster R-CNN : mAP 74.3% 비교했을때 **정확도가 낮다**는것을 알 수 있다. 이를 개선하기위해 만든것이 SSD이다.
> 2-stage detector인 Faster R-CNN의 정확도와 1-stage detector의 성능을 가지는 모델
- 논문의 저자들이 요약한 **contribution**은 다음과 같다.
> 1) small convolutional Predictor filter들을 사용하여 feature map에 적용해 <br>
고정된 default bounding boxes의 category[Confidence] score 및 box offset을 예측하는 것이다. <br>
> 2) detection의 높은 정확도를 얻기 위해 다양한 크기(scale)의 feature map에서 다양한 크기(scale)의 prediction을 생성하고, aspect ratio(종횡비)별로 예측을 구분한다. <br>
> 3) 이러한 design feature들은 저해상도 input image에서도 간단한 end-to-end training과 높은 정확도로 이어져, <br>
속도와 정확도의 trade-off를 더욱 개선한다.



## 2. Core concepts
> 논문에서 소개되는 주요 개념들은 다음과 같다.
>> Default boxes and aspect ratios, Matching strategy, Loss function

### 2-1. Default boxes and aspect ratios

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/e26b7603-f5c1-4a7f-bc74-5bef6c4e44af)



- 모델에서 Prediction을 수행하는 각 feature map들마다 bounding box regression과 object detection을 수행하기 위해 해당 feature map에 3x3 predictor filter를 가지고 convolution을 하게된다. <br> 
> SSD에서는 이때 Faster-RCNN 에서 region proposal을 위해 사용했던 앵커(Anchor)박스와 유사한 개념인 **default box**를 사용하게 된다. <br>

- 각 feature map마다 어떻게 몇개의 bounding box를 뽑는지에 대해 살펴보자. <br>
> 논문에서는 각 feature map에 차례대로 4개,6개,6개,6개,4개,4개의 default box들을 먼저 선정했다. 


<center><img src='https://drive.google.com/uc?export=download&id=1ccmUxcdZQEEW8DWiJJD8ZLmr1kgi3-B7' height="300" width="900"> </center>

> - 그 중 두번째의 19x19 Con7 피쳐맵에 대해서 bounding box를 생성하는 과정을 하나의 예제로 살펴보면 <br> 
이 feature map에서는 6개의 default bounding box를 만들고 이 박스와 대응되는 자리에서 <br>
예측되는 박스의 offset과 class score를 예측한다. <br> 
이것을 선정한 default box의 갯수만큼 반복하여 다양한 object를 찾아낸다. 즉,<br>
>> __6(default box) X (20개:Classes-PASCAL VOC기준 + 1:object인지 아닌지)=21+4(offset 좌표)) = 150(ch)__ <br>
이것이 default boxes(바운딩박스들)이 담고있는 정보이다.

<center> <img src='https://drive.google.com/uc?export=download&id=1DBgcAatbTaqWlvll6BVHDflVCgaXVCDe' height="470" width="900"> </center>

> 이를 모든 feature map 6개의 로케이션에 대해서 뽑아내면 논문에서 소개된 <br>
38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 <br>
= 총 8732개의 바운딩박스를 뽑는 과정이다. <br>

#### + Matching strategy

훈련하는 동안 어떤 default box들이 ground truth 탐지에 해당하는지 결정하고 그에 따라 네트워크를 훈련해야 합니다.

- 각각의 ground truth box에 대해서, 위치, 종횡비 및 배율이 다른 default box들을 선택

- 각 ground truth box를 최상의 자카드 오버랩이 있는 default box와 매칭시키는 것으로 시작

- 임계값(0.5)보다 높은 jaccard 오버랩을 사용하여 default box들을 ground truth와 매칭

이것은 학습 문제를 단순화하여 네트워크가 최대 중첩이 있는 default box만 선택하도록 요구하지 않고 multiple overlapping default box들에 대해 높은 점수를 예측할 수 있도록 합니다.

### 2-2. Choosing scales and aspect ratios for default boxes

<img src='https://drive.google.com/uc?export=download&id=1Ia8aFNp1HtWIQ1lf1znK4eNN-lUxpl0u' height="300" width="750">

- SSD에서 다양한 레이어의 피쳐맵들을 사용하는것은 scale variance를 다룰 수 있다. <br>
- bounding box를 찾는데에 있어 위의 그림처럼 8x8의 feature map에서는 default box가 <br> 
상대적으로 작은 물체(고양이)를 찾는데에 높은 IoU가 매칭될 것이고, <br> 
4x4 feature map에서는 상대적으로 큰 물체(강아지)에게 매칭 될 것이다. <br>
> **즉 앞쪽에 resolution이 큰 feature map에서는 작은 물체를 감지하고, <br>
뒤쪽에 resolution이 작은 feature map에서는 큰 물체를 감지할 수 있다**는 것이 <br>
multiple feature map 사용의 메인 아이디어이다. <br>

- 그렇다면 default boxes들은 어떻게 다양하게 만들어주는지 살펴볼 수 있다.


> $S_k = S_{min} + \frac{S_{max} - S_{min}}{m-1}(k-1), k ∈[1,m]$ <br>
> $ S_{min} = 0.2, S_{max} = 0.9 $
* $S_k$ = scale
* $m$ = feature map의 갯수
* $k$ = feature map index

- 위의 식으로 m개의 feature map에서 bounding box를 뽑아낸다. <br>
- 각 k값을 차례대로 넣어보면 PASCAL VOC 기준으로 <br>
$S_k = 0.1, 0.2, 0.375, 0.55, 0.725, 0.9$
라는 scale을 얻을 수 있다. <br>
> $S_0 = 0.1$ (chapter3.1의 PASCAL VOC 2007에서 conv4_3의 scale을 0.1로 setting)

- 이것은 전체 Input 이미지에서의 각 비율을 의미한다. <br>
> 즉 Input이 300x300 이미지이기 때문에 0.1은 30픽셀, 0.2는 60픽셀,각각 <br>
$ 30,60,112.5,165,217.5,270 pixels $ <br>
이와 같이 인풋 이미지를 대변한다. <br>

- 이렇게 scale이 정해지면 아래의 식으로 default box(bounding box)의 크기가 정해진다. <br>
> $a_r ∈$ {$1,2,3,\frac{1}{2},\frac{1}{3} $} <br>
$w^a_k = S_k \sqrt{a_r} $ <br>
$h^a_k = S_k /\sqrt{a_r} $ <br><br>


* For the aspect ratio of 1, we also add a default box whose scale is $S'_k = \sqrt{S_kS_k+1}$, resulting in 6 default boxes per feature map location.
> aspect ratio가 1인 경우 scale이 $S'_k = \sqrt{S_kS_k+1}$인  default box도 추가하여 feature map location당 6개의 default box가 생성됩니다.


* 아래의 식으로 default box의 중심점을 구할 수 있다.
> $(\displaystyle \frac{i+0.5}{|f_k|},\frac{j+0.5}{|f_k|})$ <br>
$|f_k|$ = k번째 feature map의 가로세로 크기 , $i,j ∈ [0,|f_k|]$ <br>


<img src='https://drive.google.com/uc?export=download&id=1lw5j-qzf7cch7-GTpVDNpRl3W_OO22h2' height="250" width="700">


* In practice, one can also design a distribution of default boxes to best fit a specific dataset. How to design the optimal tiling is an open question as well.

논문 3.1 PASCAL VOC2007에서 논문에서 테스트한 SSD 모델의 디폴트 박스에 대한 설정이 설명되어 있다.\
For conv4_3, conv10_2 and conv11_2, we only associate 4 default boxes at each feature map location – omitting aspect ratios of 1/3 and 3. For all other layers, we put 6 default boxes as described in Sec.

### 2-3. Loss function
- training 목표 : mutiple object detection

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/05433d2c-6b53-4e38-82df-e800da00ca22)


#### 0) **SSD training 목표는 MultiBox에서 파생되었으나, multiple object categories를 다루는것이 확장되었다.**
- MultiBox의 Loss식은 다음과 같다.
> $F_{conf}(x,c)=-\displaystyle\sum_{i,j}x_{ij}log(c_i)-\displaystyle\sum_{i}(1-\displaystyle\sum_jx_{ij})log(1-c_i)$
> - binary loss와 유사
> - $x_{ij} = $1 or 0
> - **IoU가 가장 높은 box만 가져와서** $\sum x_{ij}=1$이지만
- SSD는 IoU가 가장 높은 box뿐만 아니라 <br>
jaccard overlap(중복되는 부분)이 **thredhold(0.5)이상이면 모두 default box들**로 보기때문에 $\sum x^{p}_{ij}\geq1$이 된다.
> 그래서 전체 loss에서 default box들의 개수인 **$N$으로 나눠주는 이유**이다.

#### 1) $L_{conf}(x,c)$(cross entropy)

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/6b2bae50-e7c6-4089-8958-873a85718cb0)


- $x^p$$_{ij}$값은 물체의 j번째 ground truth와 i번째 default box간의 IOU가 0.5 이상이면 1, 0.5미만이면 0을 대입해준다. <br>
따라서 물체의 j번째 ground truth와 i번째 default box 간의 IOU가0.5 미만인 default box는 식에서 0이 되어 사라지게된다.

**$\Rightarrow$ 너무많은 default box가 있다. 그래서 back ground를 가리키고 있는  default box를 날려서 default box의 수를 최소화하는 작업이다.**

#### 2) $L_{loc}(x,l,g)$

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/36e60f71-dd74-4afc-928e-d0b30424ec04)


![image](https://github.com/KKH0228/Paper-Research/assets/166888979/fb50b63c-addf-46c4-9149-f0aff542dc56)



![image](https://github.com/KKH0228/Paper-Research/assets/166888979/f1858117-5743-4386-9f7a-e46a558d7ff7)


#### 3) $L_{loc}$(x,l,g)식에 나오는 용어정리
* x : Conv Predictor Filter에 의해 생성된 각 Feature Map의 Grid Cell
* x<sub>i<sub> : x에 포함되어는 특정 디폴트 박스의 offset과 class score.
* $x^{p}_{ij}$ : 

* N : ground truth와 매치된 default box의 개수
* l : predicted box (예측된 상자)
* g : ground truth box
* d : default box
* cx, cy : 그 박스의 x, y좌표
* w, h : 그 박스의 width, heigth
* $\alpha$ 에는 1을 대입해준다. 
* ground truth 와 default box의 IOU값이 0.5이상인것만 고려한다.<br>
=background object인 negative sample 에 대해서는 Localization Loss를 계산하지 않는다.


## 3. Hard negative mining

- 사실 매칭 스텝 후에 생성된 대부분의 default boxes(bounding boxes)은 object가 없는 background이다. <br>
이것은 Faster R-CNN에서도 대두되었던 문제인데 negative인, 즉 background들을 사용해 트레이닝하는것은 모델에 도움이 되지않는다. <br>
Faster R-CNN에서는 256개의 미니배치에서 128개의 positive, 128개의 negative를 사용해서 훈련하도록 권고했으나 <br> 
negative가 너무 많으면 이 비율은 어쩔수없이 유지하기 힘들다. <br>
> SDD에서는 이러한 positive와 negative의 inbalance문제에 대해 confidence loss가 높은 negative값들을 뽑아 <br> 
> **positive와 negative를 1:3 비율**로 사용하길 제안했다. 이것으로 더 빠른 최적화와 안정적인 트레이닝을 이끌었다.


## 4. Data augmentation
- 전체 원본 input 이미지를 사용한다.
- object와 jaccard IoU가 최소인 0.1, 0.3, 0.5, 0.7, 0.9이 되도록 패치를 샘플링한다.
- 랜덤하게 패치를 샘플링한다.
> 샘플링시 $\frac{1}{2}$와 $2$사이로 랜덤하게 정한다.

## 5. Experimental Results

### 5-1. Base Network
- 논문에서의 실험은 모두 **VGG16**을 기반으로 train하는데 layer를 다음과 같이  수정한다. <br>

<img src='https://drive.google.com/uc?export=download&id=1t550Gj4iQvYho2AuXBbsmCfXeK_5Ken5' height="400" width="1000">


1. VGG16 내 full connected layer 6,7을 subsampling parameter를 생성하는 **convolutional layer**로 변환한다.
2. pool5 layer(linear)를 $2$x$2$-s2, $3$x$3$-s1의 *atrous 알고리즘*을 사용하여 pool5 layer를 바꾼다.
> 일반 VGG16을 사용하는 경우 pool5를 2×2-s2 fc6 및 fc7의 subsampling parameter가 아니라 예측을 위해 conv5 3을 추가하면 결과는 거의 같지만 속도는 약 20% 느려진다.<br>
> **즉, *atrous 알고리즘*이 더 빠르다.**
3. drop out과 full connected layer 8을 없앤다.


#### + *A'trous 알고리즘 (dilated convolution)* <br>
<img src='https://drive.google.com/uc?export=download&id=12ey06VHMA4uSykBaIyXRDmDmPUJ-tVeJ' height="300" width="300"> <br>
- 기존 컨볼루션 필터가 수용하는 픽셀 사이에 간격을 둔 형태이다. <br>
입력 픽셀 수는 동일하지만, 더 넓은 범위에 대한 입력을 수용할 수 있게 된다.
- 즉, convolutional layer에 또 다른 parameter인 **dilation rate**를 도입했다.<br> 
> dilation rate는 **커널 사이의 간격**을 정의한다. <br>
> dilation rate가 2인 3x3 커널은 9개의 parameter를 사용하면서, 5x5 커널과 동일한 view를 가지게 된다.
- 적은 계산 비용으로 Receptive Field 를 늘리는 방법이라고 할 수 있다. <br>
이 A'trous 알고리즘은 필터 내부에 zero padding 을 추가해서 강제로 Receptive Field 를 늘리게 되는데, <br>
위 그림에서 진한 파란색 부분만 weight가 있고 나머지 부분은 0으로 채워지게 된다. <br>
이 Receptive Field는 필터가 한번 보는 영역으로 사진의 Feature를 파악하고, 추출하기 위해서는 넓은 Receptive Field 를 사용하는 것이 좋다. <br>
dimension 손실이 적고, 대부분의 weight가 0이기 때문에 연산의 효율이 좋다. <br>
공간적 특징을 유지하는 Segmentation에서 주로 사용되는 이유이다. 

#### Classifier : Conv 3X3X(4X(classes+4))가 나오는 이유,방식
> * **same conv 연산**하기때문
<img src='https://drive.google.com/uc?export=download&id=1JBAXfe-lbOpO_IzBRAXlA7_pHvfiVKWZ' height="300" width="750">
> * 결과 : 38x38x(4X(classes+4)) <br>
> * 19x19x(6X(classes+4)), 10x10x(6X(classes+4)), 5x5x(6X(classes+4)), <br>
3x3x(4X(classes+4)), 1x1x(4X(classes+4))도 위와 같은 방식

### 5-2. More default box shapes is better.
>- 기본적으로 6개의 default box를 사용하지만 aspect ratio=$\displaystyle \frac{1}{3}, 3$을 제거하면 성능이 0.6% 감소한다.
>- 다양한 default box shape을 사용하면 네트워크에서 box를 예측
하는 작업이 더 쉬워진다는 것을 알 수 있다.

## 6. Conclusions
- 핵심은 top layer에서 multiple feature map이 연결된 다양한 크기의 convolutional bounding box를 출력한다는 것이다.
- 가능한 box shape의 공간을 효율적으로 모델링할 수 있다.
> 단점 : 작은 feature map에서 큰 object만 detection한다.

---
Reference

[참고1](https://csm-kr.tistory.com/4)
[참고2:atrous algorithm](https://eehoeskrap.tistory.com/431)

[paper](https://arxiv.org/pdf/1512.02325.pdf) 
[1](https://csm-kr.tistory.com/4) 
[2](https://eehoeskrap.tistory.com/431) 
[3](http://www.okzartpedia.com/wordpress/index.php/2020/07/16/ssd-single-shot-detector/) 
[4](https://ys-cs17.tistory.com/12)
