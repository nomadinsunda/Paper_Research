# Fast R-CNN

https://arxiv.org/pdf/1504.08083.pdf <br>
__Ross Girshick__ <br>
Microsoft Research <br>


## 1. Introduction

SPPNet (Spatial Pyramid Pooling Network)은 기존 R-CNN이 proposal method (Selective Search)로 찾아낸 대략 2K의 RoI (Region of Interest)들에 대해서 모두 Convolution 연산하는 문제점을, Input 이미지에 대해 한번만 Convolution을 수행하고, 이 결과인 피쳐맵을 공유하는 방식으로 해결했다. <br>
* 그러나 SPPNet도 여전히 모델을 학습하기 위해서는 여러 단계를 거쳐야 했고, 학습 시에는 Fully Connected Layer만 학습시킬 수 있는 한계점이 존재했습니다. <br> 
* 이러한 한계를 극복하기 위해 Fast R-CNN이 제안되었습니다. Fast R-CNN은 SPPNet과 마찬가지로 한 번의 Convolution 연산을 수행하고, RoI pooling을 사용하여 각 RoI 영역을 고정된 크기의 피쳐 맵으로 변환합니다. 그러나 Fast R-CNN은 추가적인 Region of Interest (RoI) 추출 단계를 도입하여, CNN을 통과시킬 이미지 영역을 효율적으로 선별합니다. 이를 통해 학습 및 추론 과정을 단순화하고 속도를 향상시킵니다. 또한, Fast R-CNN은 모든 네트워크 계층을 역전파로 학습할 수 있는 end-to-end 방식을 도입하여, 이전 모델들보다 학습 및 성능을 개선했습니다.

> __"CNN feature extract부터 object classification, box regression까지 end-to-end로 하나의 모델에서 학습"__ <br>
하는것이 Fast R-CNN의 핵심 아이디어이다.

## 2. Orders of algorithm

<img src='https://drive.google.com/uc?export=download&id=1PTEChtWhkgvGJIlFZBghiktCzkMXMs90' height="300" width="400">
<img src='https://drive.google.com/uc?export=download&id=1axJgtcqc6ol1TZtecl8nf3ewvzfidXry' height="300" width="370">

Fast R-CNN의 순서는 다음과 같다.

1. input이미지를 pre-trained된 CNN에 통과시켜 피쳐맵을 추출한다.

2. Selective Search를 통해서 찾은 각각의 RoI에 대하여 RoI Pooling을 진행한다. 그 결과로 고정된 크기의 RoI feature vector를 얻는다.

3. feature vector는 fully connected layer들을 통과한 뒤, 두 개의 브랜치로 나뉘게 된다.

4. 하나의 브랜치는 softmax를 통과하여 해당 RoI를 Classification한다.

5. bouding box regression을 통해서 selective search로 찾은 RoI박스의 위치를 조정한다.

CNN을 한번만 통과시킨 뒤, 그 피쳐맵을 공유하는 것은 이미 SPP Net에서 제안된 방법이다. <br> 
그 이후의 스텝들은 SVM대신 softmax를 사용했다는 것 외에 SPPNet과 크게 다르지 않다. <br>
> Fast R-CNN의 가장 큰 특징은 이 과정들을 한데묶어 end-to-end로 엮었다는데 있다.<br> 
그 결과로 학습 속도, 성능 모두를 개선하고 CNN까지 학습할 수 있다.


 


## 3. Core concepts

논문에서 소개되는 주요 개념들을 살펴보겠습니다. 

#### 3-1. RoI Pooling concept

<img src='https://drive.google.com/uc?export=download&id=1ngn1SUYGmONsxJ9CSxy7mkuqYgcj7PMA' height="170" width="500">

* 추출된 RoI feature를 Fully connected layer에 통과시키기위해선 <br>
같은 size의 input들을  넣어줘야합니다. (각 Dense layer들과 연결하기위해) <br>
* 하지만 Selective search로 찾은 RoI들의 크기는 제각각이기때문에 <br> 이때 RoI Pooling을 통해 일정한 사이즈로 맞춰줄 수 있습니다. <br>
* 논문에서는 SPPNet의 Pyramid 레벨 1에 해당한다고 언급되었습니다. <br>


예를들어 RoI Pooling 개념을 좀 더 자세히 알아보자.

##### 1) Sampling RoI

<img src='https://drive.google.com/uc?export=download&id=1KldD8FMUd2LkEA06KbIhtz-mHGG8q96G' height="250" width="700">

* 다음과 같이 512x512 인풋 이미지가 있을때 사전학습된 VGG16 CNN 모델을 통과시키면 <br> 
모델 내의 conv레이어+pooling레이어들(reduction factors=32)을 거쳐 16x16 피쳐맵이 나오게된다. 

##### 2) Getting RoI from feature map

<img src='https://drive.google.com/uc?export=download&id=1Dfon11e-GzIQp5Cl_tbuvdza7EKccN6i' height="250" width="700">

* Selective search로 찾은 RoI박스는 매우 많지만 예시를 위해 서로 다른 4개의 박스만 표시해보았다.<br> (RoI박스는 bounding box가 아니다. proposal된 아직 후보 박스이다.) <br> <font color="red">이 RoI박스들은 CNN을 통과해 출력된 피쳐맵의 박스와 매핑될수 있어야한다.</font> <br>
RoI박스는 원래 좌표와 크기가 있는데 그것 중 하나를 살펴보자.

##### 3) Quantization of coordinates on the feature map

<img src='https://drive.google.com/uc?export=download&id=1LX0Z3kfuXPxC-ztdckDIHbRk-dWUl1xK' height="280" width="300"> ------->  <img src='https://drive.google.com/uc?export=download&id=1dl7_xTyu3t4siuYjF8BYOvaChm2DuJzy' height="280" width="300"> -------> <img src='https://drive.google.com/uc?export=download&id=1gCzvGI171caootEhyeJVweGt_X8yamQs' height="280" width="300">

* 원래 RoI박스의 크기는 145x200이고 왼쪽 모서리좌표는 192x296이다. <br>
이것을 reduction factor(32)로 나누면
> * 너비: 200/32 = 6.25 <br>
> * 높이: 145/32 = ~4.53 <br>
> * x: 296/32 = 9.25 <br>
> * y: 192/32 = 6 <br>
와 같이 계산된다. \
너비와 높이를 cell에 걸치지 않게 6과 4로 정수스케일링 해줄 수 있다. <br>
x,y좌표도 똑같이 매핑 가능하다.  <br>
* 이것들을 Quantization(정량화)라고 하다. <br>
> (이 과정에서 파란색 영역의 데이터 손실과 녹색영역의 노이즈가 추가된다.) 



##### 4) RoI Pooling


RoI가 피쳐맵에 매핑되면 이제 Max Pooling을 적용할 수 있다.

<img src='https://drive.google.com/uc?export=download&id=1KbZ4BEd_b87eJXp8BHHTOA9furPFvIta' height="250" width="600">

* 동일한 크기의 feature vector로 출력하기위해 풀링 사이즈를 설정하고(여기서는 3x3x512) <br> 현재 매핑된 RoI는 4x6x512 이기 때문에 각각 나누어주고 스케일링해주면 1x2 vector가 된다.


<img src='https://drive.google.com/uc?export=download&id=1j3TgzqOC282P20HQZC6IrYsayM5fQk7K' height="250" width="600">

* 이제 데이터를 3x3x512 행렬로 풀링 할 수 있다. <br>
Quantization(정량화) 때문에 다시 한번 전체 맨 아래 행을 잃게 된다. <br>
최상위 레이어뿐만 아니라 전체 RoI 매트릭스(512)에서 수행된다.

<img src='https://drive.google.com/uc?export=download&id=1BOHZsaBreCR9LvRfNMUXSHhuATUhncee' height="280" width="300">

Pooling을 거쳐 최종적으로 얻게된 data는 위의 주황색 부분과 같다. <br>
(기존 파란부분+풀링을 통해 하늘색 존도 사라졌다.)

#### 3-2. Multi Task loss

* 이제 인풋 이미지로부터 피쳐맵을 추출했고, <br>
피쳐맵에서 RoI들을 찾아서 RoI Pooling을 적용하여 feature vector를 구했다. <br>
* 이 벡터로 classification과 bounding box regression을 적용하여 각각의 loss를 얻어내고, <br> 이를 back propagation하여 전체 모델을 학습시키면 된다. <br>
* 이 때, classificaiton loss와 bounding box regression을 적절하게 엮어주는 것이 필요하며,<br> 이를 **Multi task loss**라고 합니다. 수식은 아래와 같다.

> $L(p,u,t^u,v) = L_{cls}(p,u) + λ[1≤u]L_{loc}(t^u,v) $

* Input <br>
> * $p = (p_0,…,p_k,p_{k+1})$ <br>
   : softmax로 얻어낸 K+1개의 확률(k개의 object, k+1은 background인지 아닌지를 나타냄) <br>
> * $u$ : 해당 RoI의 클래스 groud truth <br>
> * $t^u = (t^u_x,t^u_y,t^u_w,t^u_h)$ <br>
> : box regression을 통과한 k+1개의 클래스에 대해 x,y,w,h값을 조정하는 tk값(offset) 
> * $v$ = groundtruth의 bounding-box regression 타겟값 <br>



* $L_{cls}(p,u) = -$log$p_u$ <br>
> classification을 구하는 로스의 앞부분 <br><br>

* $L_{loc}(t^u,v) = \displaystyle \sum_{i∈\{x,y,w,h\}}$ smooth$_{L1}(t^u_i - v_i)$ <br>
> Bounding Box regression을 통해 얻는 로스의 뒷부분 <br>


> * 인풋은 정답과 매칭되는 bounding box regression 예측 값($t^u$)과 <br>
ground truth 타겟 값($v$)을 받는다. <br>
> * 그리고 x, y, w, h 각각에 대해서 예측 값과 정답 값의 차이를 계산한 다음, <br> smooth$L1$ 함수를 통과시킨 합을 계산한다. <br>
>> <img src='https://drive.google.com/uc?export=download&id=1hEicWUV0ehmusQDmCni8Sh9mA2U2LJOU' height="50" width="250"> <br>
>> * 예측 값과 정답 값의 차가 1보다 작으면 0.5$x^2$로 $L2$distance를 계산해준다. <br>
>> * 1보다 클 경우엔 $L1$distance를 계산해주는 것을 볼 수 있다.<br>
>> * 논문에서 테스트 중 예측값이 정답 값과 너무 차이가 많이 나는 경우, <br>
그 예외값들을 그대로 $L2$ 계산하여 적용할 시 gradient가 폭발해버리는 현상때문에 <br> 차이가 큰 값들은 따로 $L1$ 계산해주는 함수로 커스텀하였다.

## 4. Conclusion

- 의의 : Fast R-CNN은 object detection 작업을 end-to-end 모델로 제시하면서 <br>
모델을 단일하게 학습시키고, 세부 단계를 간소화하며 정확도와 성능을 향상시켰습니다. <br>

- 단점 : 아직도 region proposal method로 selective search(CPU 사용)를 사용하여 속도와 효율성 면에서 제한이 있습니다. <br>
- 후에 소개되는 Faster R-CNN은 Fast R-CNN의 구조를 유지하면서 <br>selective search를 탈피하고 Anchor box 개념을 도입하여 <br>Region Proposal을 네트워크 내부로 통합했습니다. 이를 통해 더욱 효율적인 객체 검출이 가능해졌습니다. <br>
> Faster R-CNN 논문을 통해 Region Proposal Network (RPN)에 대해 자세히 알아볼 수 있습니다.


```python

```
