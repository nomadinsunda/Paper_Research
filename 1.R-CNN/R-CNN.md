# R-CNN

## 0. Object Detection이란?
> 여러 물체에 대해 어떤 물체인지 분류하는 Classification 문제와 그 물체가 어디 있는지 박스를 통해 (Bounding box) <br>
> 위치 정보를 나타내는 Localization 문제를 둘 다 해내야 하는 분야를 뜻합니다.

## 1. 정의

<center><img src='https://drive.google.com/uc?export=download&id=1miKyYHhxKqWZY2cVbZWUjGt4CdQCU0uP' height="400" width="600">

Figure 1 : R-CNN </center>


```python

```

- classification을 수행하는 CNN과 localization을 위한 regional proposal을 연결한 two-stage detector 모델 <br>

* * regional proposal로 **selective search** 사용

<center><img src='https://drive.google.com/uc?export=download&id=1h90Su4DrnyjQ0LROG0OCjB-dmDYnZxiS' height="200" width="750"> </center>

<center> Figure 1-1 : two-stage detector </center>

> * **two-stage detector** : R-CNN, Fast R-CNN, Faster R-CNN

<center><img src='https://drive.google.com/uc?export=download&id=13LMM-SzahExBRKw3Yj69dRjq95zZrzDj' height="200" width="750"> </center>

<center> Figure 1-2 : one-stage detector </center>

> * **one-stage detector** : regional proposal과 classification이 동시에 이루어지는 것
>* Ex) YOLO, SSD
* classification으로 object가 box내에 있는지,판단하고 Bounding box regressor를 이용하여 Bounding box를 찾아낸다.

### 1-1. regional propsal이란? <br>
- object가 있을만한 영역을 추출
- Ex) selective search, regional propsal Network(RPN)

#### 1) **selective search란?** <br>
- Color, 무늬(Texture), 크기(Size), 형태(Shape)에 따라 유사한 Region을 계층적 그룹핑 방법으로 계산한뒤 bounding box를 그리는 기법

<center><img src='https://drive.google.com/uc?export=download&id=1ZSNde4zWAl4zPJGNp_oWWVKGsaHxXC8X' height="350" width="500">

Figure 2 : selective search </center>

#### 2) RPN는 이후 *Faster R-CNN* 참고

### 1-2. classification란?
- CNN을 통해 추출한 벡터를 가지고 각각의 클래스 별로 SVM Classifier를 학습시킨다.
> Figure 1 참고

## 2. 프로세스

<center><img src='https://drive.google.com/uc?export=download&id=1No4HNqF7OKo0d5UKsAKgon_i6ZFNZVx0' height="250" width="800">

 Figure 3: : R-CNN : Region with CNN features </center>

* 1) Input image에 selective search를 통해 regional proposal output(RoI라 통칭) 2,000개를 추출하여 복사한다. <br>
> 이과정을 croping이라 한다.

* 2) 2,000개의 RoI를 CNN모델에 넣기 위해 w, b를 동일한 size로 맞춘다.
> 이렇게 데이터를 동일한 size로 바꾸는 작업을 warping이라 한다. <br>
>> Q. 왜 동일한 size로 만들까? <br>
 - Convolution Layer에 input size가 고정적이지않고 FC layer에서의 input size는 고정이므로 <br>
 Convolution Layer에 대한 output size도 동일한 size로 줘서 output size를 동일하게 하려는것

* 3) 2,000개의 wraped image를 CNN모델에 넣는다.

* 4) 각각의 CNN 결과값을 classification해서 object가 bounding box내에 있는지 판별하여 결과를 얻는다.

## 3. Bounding-Box regression

### 3-1. 정의

- ${(P^i, G^i)}_{i=1, ...,N}$ : $ N $ 개의 training set

- $P^i = (P_x^i,P_y^i,P_w^i,P_h^i) $ <br>
: $P^i$는 **RoI 예측 box인 $P^i$**의 중심좌표값($x,y$)과 $width$와 $height$을 지정한다.

- $G = (G_x,G_y,G_w,G_h) $ <br>
: $G^i$는 같은 방법으로 **ground-truth인 bounding box**를 지정한다.

> Figure 4에서는 소문자로 표기한다.


#### + **ground truth란?**
- 학습하고자 하는 데이터의 원본 혹은 실제 값

### 3-2. 목표
- Roi 예측 box인 $P$를 ground-trouth $G$에 매핑해서 변환을 학습하는 것이다. ($P$가 $G$에 유사하게 되도록 만드는것)
> 따라서 Roi 예측 box인 $P$와 Ground truth의 Bounding box $G$의 차이( Loss값)가 최소한이 되도록 학습된다.




### 3-3. 과정

<center><img src='https://drive.google.com/uc?export=download&id=1le22aslxMu6duhLy6hXqvj0Kg2jTyC7l' height="350" width="500">

 Figure 4: Illustration of transformation between predicted and ground truth bounding boxes </center>

#### 1) $ d_\star(P) $ 란?
- input $P$를 예측된 ground truth box인 $\hat{G}$에 transformation시키는 변수
> $\hat{G}_x = P_wd_x(P) + P_x$ <br>
> $\hat{G}_y = P_hd_y(P) + P_y$ <br>
> $ \hat{G}_w = P_w $ exp $(d_w(P)) $ <br>
> $ \hat{G}_h = P_h $ exp $(d_h(P)) $ <br>
> - $x, y$는 좌표값으로 위치만 변형시켜주면 되기때문에 선형함수를 썼지만
> - $w, h$는 이미지의 비율에 맞춰서 변형시켜야 하기때문에 exp함수를 사용했다.

*  $ w^{T}_\star $를 업데이트해서 $ d_\star(P) $를 얻는것이 목표<br>
 > (1) $ d_\star $ 함수를 구하기전에 CNN을 통과할 때 pooling 5 layer서 얻어낸 feature vector(map)를 사용하는데
>> 이것을 $ \phi_5 (P) $라고 정의한다. <br>

 > (2) 그리고 함수에 학습 가능한 weight 벡터($ w^{T}_\star $)를 주어 계산한다. <br>
>> $d_\star(P) = w^{T}_\star \phi_5 (P) $



* $ w^{T}_\star $는 *Ridge regression*으로 정규화해서 weight를 optimizer

<center><img src='https://drive.google.com/uc?export=download&id=1JIlkIJyQDI5AfFfSFiufTYh7WBrba0tC' height="100" width="500">

Figure 5: $ w^{T}_\star $ 정규화 (Ridge regression 사용) </center>




<center><img src='https://drive.google.com/uc?export=download&id=1tc_1xUk50sMyafH5euSI-LB3AZACOqLL' height="500" width="850">

> **릿지 회귀(Ridge Regression)란?**
- 평균제곱오차(MSE)를 최소화하면서 $P $의 계수인 $w_\star$의 L2 norm을 제한하는 기법
- 하이퍼파라미터 λ가 클수록 β의 L2 norm이 줄어든다 (λ는 3-4에서 설명)


#### 2) $t_\star$ 란?
- $ (P,G) $의 training의 regression targets
- $P$를 $G$로 이동하기위해 필요한 이동량

> $ t_x = (G_x-P_x) / P_w $ <br>
> $ t_y = (G_y-P_y) / P_h $ <br>
> $ t_w = log(G_w / P_w) $ <br>
> $ t_h = log(G_h / P_h) $ <br>

### 3-4. Bounding-Box regression Point
- 1) 정규화가 중요하다.
> validation set를 기반으로 λ = 1000으로 설정한다.

- 2) 사용할 training 쌍 $(P,G)$을 선택할 때 주의를 기울여야 한다.
> 직관적으로 P가 모든 ground-truth box들과 **거리가 멀다면** P를 ground-truth box G로 변환하는 작업은 **의미없다.** <br>
P와 같은 예를 사용하면 희망 없는 학습 문제가 발생하기 때문에 <br> **proposal P가 적어도 하나의 ground-truth box 근처에 있는 경우**에만 학습한다.<br>

>>overlap 부분이 threshold(validation set에선 0.6 사용)보다 큰경우에만 최대 IoUoverlap(둘이상 중첩되는 경우)갖는 ground-truth box G에 P를 할당하여 "근접성"을 구현하고 할당되지 않은 모든 proposal은 삭제한다. <br>
>>test할때 각 proposal에 점수를 매기고 새로운 detection window를 한번만 예측한다.
>> 이 과정을 반복할 수 있지만 결과가 개선되지 않는다.

## 4. 단점
* RCNN은 selective search를 통해 찾은 각각의 약 2,000개의 Region of Interest(ROI) 영역에 대해 개별적으로 CNN 연산을 수행해야 합니다. 이는 매우 많은 연산량과 시간이 소요됩니다.
* 속도 저하의 가장 큰 병목 구간은 selective search를 통해서 찾은 2,000개의 영역에 모두 CNN inference를 진행하기 때문이다.
> SPPNet은 이러한 RCNN의 단점을 보완하기 위해 도입되었습니다. SPPNet은 입력 이미지에 대해 단 한 번의 CNN 연산을 수행한 후, 그 결과물에서 여러 개의 ROI 영역에 대해 고정된 크기의 feature를 추출합니다. 이로써 매번 ROI마다 CNN 연산을 수행하는 비효율성을 해결하고, 전체적인 속도를 향상시킬 수 있습니다. 이러한 방식으로 SPPNet은 RCNN의 속도 저하 문제를 완화시키면서도 정확성을 유지합니다.

---
# Reference

[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

[bounding box regression](https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/)
