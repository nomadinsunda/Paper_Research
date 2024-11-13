# Retina-net (Focal Loss for Dense Object Detection)

## 1. Introduction
- state-of-art object detector은 two-stage를 기반으로 하는데 <br>
one-stage로 two-stage만큼의 정확성을 높일수 있을까?하는 질문에서 시작된다.<br>
- 또한 Object detection 모델은 이미지 내의 객체의 영역을 추정하고 IoU threshold에 따라 <br>
positive(object)와 negative(background) sample로 구분한 후, 이를 활용하여 학습한다.
- 하지만 일반적으로 이미지 내 object의 수가 적기 때문에 <br> **positive sample(object, foreground)**은 negative sample(background)에 비해 **매우 적다.** <br>
그래서 둘사이 큰차이가 생겨서 **class imbalance**가 발생한다.

### class imbalance
- R-CNN처럼 two-stage cascade, sampling heuristic으로 해결된다.
- region proposal 단계(예:Selective search, Edgeboxes, Deepmask, RPN)에서 <br>
object 후보군들의 수를 줄였으나 대부분의 background를 필터링한다(잔존시킨다).
- 다음 단계인 classification에서는 foreground과 background 비율을 1:3로 설정(hard negative mining)하거나 online hard example mining(OHEM)처럼 sampling heuristic은 foreground와 background사이 균형을 유지하기위해 진행된다.
- 대조적으로 one-stage detector는 이미지 전체에서 sampling되는 많은 object 후보군들을 처리해야한다.
- 비슷한 sampling heuristic이 적용될 수도 있지만 여전히 쉽게 분류되는 background에 의해 training 절차가 지배되기 때문에 비효율적이다. 
- 이러한 비효율성은 일반적으로 bootstrapping과 hard example mining 같은 기술을 통해 해결되는 object detection의 고전적인 문제이다.
> 이 논문에서는 class imbalance를 다루는 이전의 접근방식보다 더 효과적인 **새로운 loss function**을 제안한다.<br>
> loss function는 동적으로 스케일링된 cross entropy loss이며, 올바른  class(object)에 대한 신뢰도가 증가함에 따라 스케일링 계수가 0으로 감소한다. <br>
> 직관적으로 스케일링 계수는 trainig하는동안 easy(쉽게 background라고 예측할수있는) example의 기여도를 down-weight(가중치를 낮춰주고)하고 hard(object가 있다고 예측할수있는) example에 중점을 둔다.

- **Focal loss**
> 높은 정확성, hard example mining과 sampling heuristic의 대안을 능가하는 one-stage detector<br>
- **RetinaNet**
>피처맵 피라미드와 앵커박스 사용을 특징으로, ResNet-101-FPN backbone을 기반으로 효율과 정확성이 높은 모델이다.



## 2. Focal Loss
- 정의 : training동안 foreground와 background사이 매우 높은 imbalance를 해결하기위한 one-stage object dection (ex: 1:1000)

### 2-1. cross entropy (CE)
- focal loss는 binary classification인 cross entropy에서 출발한다.<br>
> $y=1$ : 앵커박스에 object가 있을때 확률 => $p$ <br>
> $ CP(p,y) = 
  \begin{cases}
    -log(p),         \;if\; y=1 \\
    -log(1-p),       otherwise.
  \end{cases}$ <br>
> $y \in \{-1, 1\} $ : groundtruth 값 <br>
> $p \in [0, 1]$ : 앵커박스안에 object가 class 객체인지 아닌지 확률

- $y=1$일때 $p_t$를 다음과 같이 정의할 수 있다.
> $ p_t = 
  \begin{cases}
    p,      \;if\; y=1\\
    1-p,    otherwise.
  \end{cases}$
>> 그래서 $CE(p,y)=CE(p_t)=-log(p_t)$이다.

 - CE loss는 아래 Figure 1의 그래프에서 파란색 곡선이다.<br>
![figure1](https://github.com/KKH0228/Paper-Research/assets/166888979/002afe3d-caba-4753-8046-05f5aa70a286)


 - 이 loss의 주목할만한 특징은 plot에서 보여지는데 쉽게 분류되는 ($p_t(IoU)$ >=0.5일때) <br>
easy example(물체가 있는지, 없는지 확신할수있는 확률이 높은 example)들도 <br> 사소하지 않은(무시할 수 없는) 규모의 loss를 초래한다.<br>
> easy example들의 많은 수의 loss들를 합하면, <br>
우리가 실제로 판별해야할 rare class(object인지 아닌지 애매한 class)들을 압도하는 loss가 될 수 있어서 <br>
training하고싶은 loss들을 training할 수 없고 학습 효율이 떨어진다.

### 2-2. Balanced Cross Entropy
- class imbalance를 해결하기 위한 일반적인 방법은 **weighting factor($\alpha$)를 넣는 것**이다.
- 이것은 positive와 negative의 중요도를 따질 수 있게 된다.
> $\alpha \in [0, 1]$ : class 1(=foreground, object가 있을때, $y=1$) 일때 <br>
> $1- \alpha $ : class-1(=background, object가 없을때, $y=o.w$) 일때 

- 실제로 α는 inverse class 빈도에 의해 설정되거나 cross-validation에 의해 설정되는 hyperparameter로 처리될 수 있다.
- 표기상의 편의를 위해 $p_t$를 정의한 방법과 유사하게 $α_t$를 정의한다. α-balanced CE loss를 다음과 같이 쓴다.
> $CE(p_t) = -\alpha_t log(p_t)$
- 하지만 여전히 easy example과 hard example에 대한 구분은 하지않는다.



### 2-3. Focal Loss Definition
- 실험에서 알 수 있는것처럼, 큰 imbalance는 많은(dense) detectoer를 training하는 동안에 <br>
cross-entropy의 loss를 압도해서 쉽게 분류되는 negative(background)가 대부분의 loss와 gradient를 차지한다.
- $\alpha$가 positive/negative인지(object가 있는지 없는지) balance하는동안 <br>
easy/hard(foreground인지, background인지) example을 구분하지 않는다.
- 대신에 우리는 easy example(쉽게 background임을 구분하는)에 down-weight하고 hard negative에 집중하는 loss function을 재구성한다.
> down-weight : 가중치를 낮춰서 학습의 비중을 낮춘다는 의미
- 공식적으로 **modulating factor인 $(1-p_t)^{\gamma}$와 tunable focusing parameter $\gamma\geq0$**를 cross entropy에 추가한다.
> $FL(p_t)=-(1-p_t)^{\gamma}log(p_t)$

- focal loss는 $\gamma \in [0,5]$ 값인 ${0, 0.5, 1, 2, 5}$을 대입하여 Figure 1를 보여준다.
- focal loss는 두가지 특성이 있다.
> 1. $p_t$와 modulating factor의 관계 <br>
> - example을 잘못 분류해서 $p_t$가 매우 작을때(hard example일때), <br> 
>> modulating factor는 1에 가까워지고 loss값은 영향을 받지 않는다.
> - $p_t \rightarrow1$이면 <br>
>> modulating factor는 0에 수렴하고 잘 분류된(쉽게 background라고 확신할 확률이 높은) example들의 loss를 down-weight한다.
>2. Focusing factor $ \gamma $는 easy example을 down-weighted하는 정도를 부드럽게 조절한다. <br>
> - $ \gamma =0 $이면, Focal loss는 CE와 같고, <br>
> - $ \gamma$가 증가하면, modulating factor인 $(1-p_t)^{\gamma}$도 증가한다.
>> $ \gamma =2 $가 가장 잘 작동한다.

- 직관적으로 modulating factor은 easy example의 loss 기여도를 줄이고, example이 작은 loss를 받는 범위를 늘리는 역할을 한다. 
> -  $\gamma =2$와 example에 물체가 있다고 확신하는 확률, 물체가 있을 확률 $p_t=0.9$일때, 
>> CE에 비해 100배 적은 loss를 가지고, <br>
> - $p_t ≈ 0.968$ 일때, 
>> 1000배의 적은 loss를 가진다.<br>
- 이는 잘못 분류된 (object가 있는데 background로 예측하거나 background를 object로 예측하는) <br>
examples을 수정하는 중요도를 상승시킨다는 의미이다.
> $\gamma =2, 0.5 \geq p_t$일때 loss가 최대 4배로 줄어든다.
- 이렇게 **$p_t$에 따라 loss값을 조절하여 easy example의 영향을 낮추고 hard example의 영향을 높이는 것**이다.

### 2-4. $\alpha$-balanced variant of the focal loss
> $FL(p_t)=-\alpha(1-p_t)^{\gamma}log(p_t)$

- the non-α-balanced 형태에 비해 **약간 향상된 정확도를 제공해서** 이 공식을 채택한다. 
- 마지막으로, loss layer의 구현은 $p$를 계산하기 위한 시그모이드 연산과 loss 계산을 결합하여 <br>
더 큰 수치적 안정성을 가져온다는 점에 주목한다.


### 2-5. Class Imbalance and Model Initialization

- binary classification 모델은 기본적으로 $y = -1$ 또는 $1$을 출력할 동일한 확률을 갖도록 초기화된다. 
- 이러한 초기화에서 class imbalance이 있는 경우 빈도가 높은 class가 전체 loss을 차지하고, 초기 training에서 불안정성을 유발할 수 있다. 
- 이를 해결하기 위해 학습 시작 시 rare class(object인지 아닌지 애매한 class)에 대한 모델에서 추정한 $p$ 값에 대한 prior 개념을 도입한다. 
> prior을 π로 정의하고, 모델의 추정된 $p$가 클래스가 낮도록 설정한다. (예로 0.01)
- 이것은 loss function의 변경이 아니라 Model Initialization의 변경이다. <br>
- 이것이 heavy class imbalance의 경우 cross entropy와 focal loss 둘다 training 안정성을 향상시킨다.

### 2-6. Class Imbalance and Two-stage Detectors
-  Two-stage detectors는 종종 α-balancing이나 focal loss 없이 사용하지 않고 cross entropy loss로 training된다. <br>
대신 1) a two-stage cascade와 2) 편향된 minibatch sampling을 통해 class Imbalance을 해결한다. 
- 1) a two-stage cascade
> 무한한 세트→ 1,000 또는 2,000으로 줄이는 object proposal mechanism <br>
>> 중요한 점 : 선택된 the selected proposal는 random이 아니지만, <br>
대부분의 easy negative(쉽게 background라고 예측하는 example)를 제거하는 true object locations에 해당할 가능성이 높다는 것이다.
- 2) 편향된 minibatch sampling 
> 일반적으로 예를 들어 positive와 negative의 비율이 1:3인 minibatches를 구성하는데 사용된다. <br>
> 이 비율은 암시적 sampling을 통해 구현되는 α- balancing factor이다. <br>
> 제안된 focal loss는 loss function를 통해 직접 one-stage detection system에서 이러한 mechanism을 해결하도록 설계되었다.

## 3. RetinaNet Detector

<img src='https://drive.google.com/uc?export=download&id=1WhkpJ13lEeHrfWY-txv54KRq8x_HH7XV' height="400" width="1000"> <br>

이제 focal loss를 적용할 RetinaNet의 전체적인 구조를 살펴볼수 있다. <br>
Backbone은 Resnet을 사용하고, FPN(Feature Pyramid Networks)을 적용하였다. <br>
그리고 detect를하는 head 부분에서는 RPN head대신 자체적인 컨볼루션 head를 사용하였다. <br>

### 3-1. Backbone Part

<img src='https://drive.google.com/uc?export=download&id=1wK1RstkfX1geuWM2WjTHvgO2aK4zLqoE' height="750" width="1000"> <br>


Backbone 부분을 살펴보면 FPN과 상당히 유사함으로 [FPN논문](https://arxiv.org/pdf/1612.03144.pdf)을 참조하면 된다.<br> 
1. 인풋 이미지(그림은 640x640으로 설정)부터 차례대로 C3,C4,C5의 피쳐맵을 뽑아낸다. <br>
(컴퓨팅 리소스를 많이 차지한다고 판단해 FPN과 달리, C2는 사용하지않는다)<br>
2. 1x1 컨볼루션 연산을 통해 채널을 256으로 맞춰준다.
3. 이후 C5부터 Top-Down pathway와 lateral connection을 이용해 다양한 스케일을 커버할 수 있는 효율적인 네트워크를 구성한다. <br>
(그 다음 피쳐맵과 연결하여 semantic value를 높여서) <br>
4. 또다른 FPN과의 차이점인 P6는 C5를 기반으로 3x3 컨볼루션 다운샘플링하여 생성한다.
5. 이렇게 모두 P3 ~ P7 까지의 피쳐맵을 획득한다.

> 우측 하단에는 생성될 앵커박스의 스케일과 비율도 만들어놓았다. <br>
FPN과 다르게 각 피쳐맵마다 3개의 스케일과 3개의 비율, 각 9개의 앵커가 사용된다. <br>
가장 작은 앵커는 $32$x$2^0=32$ 이고, 가장 큰 앵커는 $512$x$2^\frac{2}{3}$ 약 $813$이다.


### 3-2. Head Part

<img src='https://drive.google.com/uc?export=download&id=1Wj-5cuDS1xu1UUGxN8j6a3FZUvoSOnAQ' height="550" width="800"> <br>

이제 생성된 P3 ~ P7의 피쳐맵에 각각 Head를 붙여줘서 detecting을 할 차례이다. <br> 
RetinaNET은 one-stage 네트워크이므로 RoI Pooling을 사용하지않고 위와같이 가중치 공유 기반의 컨볼루션 detect(predict) head를 사용한다. <br>
이것은 각각 Classification을 하는 Class subnet과 Box regression을 하는 Box subnet으로 나뉜다. <br>
즉 256채널을 각각 KA, 4(cx,cy,w,h)A로 매핑한다. <br>
A는 앵커의 수(9개)이고, K는 detect할 클래스의 수이다. 여기서는 Background를 포함하지않으며 파스칼 VOC 기준으로 20이다. <br>


### 3-3. Anchors

<img src='https://drive.google.com/uc?export=download&id=1WD13pmz-X9QjjUiIzVKO_ph8PjPHQ4pl' height="450" width="700"> <br>


RetinaNet에서는 매우 많은 앵커박스를 사용하는데 이것을 만드는 기준은 위와 같다. <br>
FPN에서는 각 피쳐맵 스케일마다 하나의 크기와 3개의 비율을 가지는 앵커박스를 사용했고, <Br> RetinaNet에서는 각 피쳐맵 마다 3개의 크기, 3개의 비율을 가지는 앵커박스 총 9개를 사용한다. <br>
예를들어 [600,600,3]의 input 이미지가 들어왔다면 P3의 크기는 $\frac{1}{8}$이 된 75x75 사이즈의 피쳐맵이 될것이다. <br>
이제 각 그리드 셀 마다 9개의 앵커가 class개의 vector, <br>
4개(박스좌표)의 vector가 각각 tiled되어 class subnet , box subnet의 정보를 구성하는 박스가 된다. <br>
학습시에는 IoU가 0.5 이상인 박스들만이 positive, <br>
0.4 이하인 박스들은 negative(background)로 할당이 된다. <br>

<img src='https://drive.google.com/uc?export=download&id=1EHU8rvbRUEpUegaW90fXEGIDFJInUEVP' height="530" width="1000"> <br>


그렇다면 [Bx600x600x3]의 input 이미지가 들어온다면 앵커박스는 총 몇개가 될까? <br>
첫번째 피쳐맵인 P3에서는 $\frac{1}{8}$로 줄어든 75x75x9(앵커) = 50625개 <br>
여기서 이미 5만개가 넘는 앵커박스를 생성하게 된다. <br>
그렇게 마지막 P7까지 더해주면 output은 [B, 67995, 80]이 될 것이다. (80=class 80개인 coco dataset)<br>
> SSD나 FPN과 달리 매우 dense한(많은) 앵커를 사용하여 정확한 객체 탐지를 노리는데, <br> 이것은 background 가중치를 줄이는 focal loss를 사용하기 때문에 가능하다.

## 4. Training & Inference

### 4-1. Inference 

* 속도향상을 위해 confidence를 0.5이상으로 threshold를 하고 제일 높은 score들만 이용 해서 box prediction을 decode하였다.
 -> 여기서 decode란 relative offset을 ground-truth 좌표로 바꾸는 과정을 말한다. 
 
* focal loss의 경우 classificataion subnet자체에 적용을 시켰다. 

* retinanet을 학습할때 focal loss는 한 이미지당 전체 앵커에 대해서 적용을 했고 따라서 전체 focal loss는 전체 앵커의 focal loss합으로 계산이 되고 ground truth box로 할당된 앵커의 개수를 normalizationn했다.

 
* 모든 수준의 top prediction이 병합되고 0.5 threshold로 NMS(non-maximum-suppression)를 수행하여  detection 결과를 얻는다.

### 4-2. initialization
* initialization을 쓰는이유는  많은 수의 background anchor가 훈련의 첫 번째 반복에서 불안정한 큰 손실 값을 생성하는 것을 방지하기 위함이다.

 -> 첫 번째 반복에서 불안정한 큰 손실 값을 생성되는 이유는 class imbalance때문이다.

* RetinaNet 서브넷의 final conv layer를 제외한 모든 새로운 convlayer는  b(bias) = 0으로 초기화되고 가우스 가중치는 σ = 0.01로 초기화 된다.

* classification subnet의 final conv latyer에 대해 bias 초기화를 b = − log((1 − π)/π)로 설정한다. 여기서 π는 훈련 시작 시 모든 앵커가 ~π의 confidence를 가진 foreground로  레이블되어야 한다. 

* 논문 실험에서는 π = 0.01을 사용한다. 

### 4-3. optimization
* stochasti gradient descent(SGD)를 사용한다.

* 달리 지정하지 않는 한 모든 모델은 초기 학습률이 0.01이고 90,000번의 반복학습을 한다. 

* 달리 명시되지 않는 한 data augmentation 의 유일한 형태로 수평 이미지 뒤집기(좌우반전) 를 한다.

* 훈련 손실: box regression에 사용된 focal loss와  standard smooth L1 loss의 합입니다.

### 4-4. Training
* resnet 50또는 resnet101을 사용했다.

* training과 test 모두 이미지크기를 600픽셀로 잡았다.

* RetinaNet의 판단 속도 향상을 위해, 각 FPN level에서 가장 box prediction 점수가 높은 1,000개의 box만 result에 사용하였다.

* RetinaNet을 COCO 데이터셋을 통해 학습시킨 후 서로 다른 loss function을 사용하여 AP 값을 측정했다.






### 4-5. Training 결과

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/721c0662-f774-4893-9065-a0b19b8053dc)


* foregound example일 경우 감마에 따른 변화가 크지않고 background example일일 경우 감마에 따른 변화가크다

* 감마가 커질수록 negative sample에 집중할수 있다.

* easy negative에 집중을 줄이고 hard negative에 집중한다.

### 4-6. Two stage와 One stage의 AP비교

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/d8b5a360-7a3b-4a75-ba5b-3779cbc5dcc6)


* RetinaNet의 성능은 기존 two-stage, one-stage 모델들과 비교해보았을 때도, 우수함을 알 수 있있다.

### 4-7. CE와 focal loss비교

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/7c42ee1b-dbfa-4f01-b0f5-30004e107768)


* Best OHEM 보다 Best Focal Loss가  AP값이 더 높다.

### 4-8. Accuracy 와 speed는 trade-off관계이다.

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/b5cf2636-16b4-4f57-8437-4ae0bc113852)

---
Image & Reference

[paper](https://arxiv.org/pdf/1708.02002.pdf) [1](https://csm-kr.tistory.com/5)

Focal Loss는 Lin 등이 2017년에 발표한 논문 "밀집 객체 검출을 위한 초점 손실(Focal Loss for Dense Object Detection)"에서 소개된 손실 함수입니다. 이 함수는 객체 검출 작업에서 클래스 불균형 문제를 해결하기 위해 설계되었습니다. 여기서 배경 예제의 수가 긍정적인 예제보다 훨씬 많은 경우가 많습니다. Focal Loss 함수는 어려운 예제(즉, 높은 신뢰도로 잘못 분류된 예제)에 더 큰 가중치를 할당하여 소수 클래스에 더 많은 관심을 기울입니다.

Focal Loss 함수는 다음과 같이 정의됩니다:

$L_{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$

여기서 $p_t$는 주어진 예제에 대한 양성 클래스의 예측 확률이고, $\alpha_t$는 손실에 양성 예제의 기여를 균형있게 조절하기 위해 양성 예제에 적용되는 가중치 요소입니다. $\gamma$는 쉬운 예제의 가중치가 감소하는 속도를 조정하는 초점 매개 변수입니다. $(1 - p_t)^\gamma$는 초점 요인으로, 쉬운 예제의 기여를 낮추고 어려운 예제의 기여를 높입니다.

$\gamma$의 값은 쉬운 예제의 가중치가 감소하는 속도를 제어합니다. $\gamma$가 0으로 설정되면 초점 손실은 표준 크로스 엔트로피 손실로 감소합니다. $\gamma$가 증가함에 따라 쉬운 예제의 가중치가 더 빨리 감소하므로 손실이 더욱 어려운 예제에 초점을 맞출 수 있습니다.

$\alpha_t$의 값은 양성 예제에 더 중요성을 부여하는 균형 요소입니다. 다음과 같이 정의됩니다:

$\alpha_t = \begin{cases}
\alpha & \text{if } y=1
1-\alpha & \text{otherwise}
\end{cases}$

여기서 $y$는 실제 클래스 레이블(양성은 1, 음성은 0)이고, $\alpha$는 양성 및 음성 예제 사이의 균형을 조절하는 하이퍼파라미터입니다. $\alpha$를 0.5보다 작은 값으로 설정하면 양성 예제에 더 많은 가중치를 부여하고, 0.5보다 큰 값으로 설정하면 음성 예제에 더 많은 가중치를 부여합니다.

전반적으로, 초점 손실 함수는 어려운 예제에 더 많은 가중치를 부여함으로써 불균형한 분류 문제에서 모델의 성능을 향상시킵니다. 이를 통해 모델은 소수 클래스에 더 많은 초점을 맞추고 전체적으로 더 나은 성능을 달성할 수 있습니다.

# 1. Introduce

현재 최첨단 물체 검출기는 두 단계(proposal-driven mechanism) 기반으로 만들어졌습니다. R-CNN 프레임워크[11]에서 처럼 첫 번째 단계에서는 후보 물체 위치의 희소한(set of candidate object locations) 집합을 생성하고 두 번째 단계에서는 각 후보 위치를 전반적인 클래스 또는 백그라운드로 분류합니다. <br><br> 이는 컨볼루션 신경망을 사용합니다. [10, 28, 20, 14]의 연구를 통해 이러한 두 단계 프레임워크는 COCO 벤치마크[21]에서 꾸준히 최고 정확도를 달성하고 있습니다.
두 단계 검출기의 성공에도 불구하고, 자연스럽게 물어볼 수 있는 질문은 간단한 단계 검출기가 유사한 정확도를 달성할 수 있는가입니다. <br><br>한 단계 검출기는 물체 위치, 스케일 및 종횡비의 정규, 조밀한 샘플링(sampling) 위에 적용됩니다. YOLO[26, 27] 및 SSD[22, 9]와 같은 단계 검출기에 대한 최근 연구는 더 빠른 검출기를 생성하면서 최첨단 두 단계 방법과 상대적으로 10-40% 정확도 내지는 유사한 결과를 제시합니다.

이 논문은 한 단계 물체 검출기를 제시하여 더 나아가 최첨단 Feature Pyramid Network (FPN)[20] 또는 Faster R-CNN[28]의 Mask R-CNN [14] 변형과 같은 복잡한 두 단계 검출기와 동등한 COCO AP를 처음으로 달성합니다. <br><br>이 결과를 얻기 위해, 우리는 훈련 중 발생하는 클래스 불균형을 한 단계 검출기가 최첨단 정확도를 달성하는 데 있어 가장 큰 장애물로 식별하고 이를 극복하기 위한 새로운 손실 함수를 제안합니다.

클래스 불균형은 R-CNN과 같은 검출기에서 두 단계의 카스케이드와 샘플링 휴리스틱으로 해결됩니다. 제안 단계 (예: Selective Search [35], EdgeBoxes [39], DeepMask [24, 25], RPN [28])는 후보 객체 위치의 수를 빠르게 줄여 대부분의 배경 샘플을 필터링하여 작은 수의 후보 객체 위치 (예: 1-2k)로 좁힙니다. <br><br>두 번째 분류 단계에서는 고정된 전경 대 배경 비율(1:3) 또는 온라인 어려운 예 채굴(OHEM) [31]과 같은 샘플링 휴리스틱을 수행하여 전경과 배경 사이의 관리 가능한 균형을 유지합니다.

한편, one-stage 검출기는 이미지 전체에서 규칙적으로 샘플링된 후보 객체 위치의 훨씬 더 많은 집합을 처리해야 합니다. 실제로는 공간 위치, 크기 및 종횡비를 밀도 있게 커버하는 약 100,000개의 위치를 열거하는 것과 같습니다. <br><br>비슷한 샘플링 휴리스틱을 적용할 수 있지만, 훈련 프로시저는 여전히 쉽게 분류되는 배경 예제에 의해 지배되기 때문에 비효율적입니다. 이 비효율성은 객체 검출에서 전형적으로 부트스트래핑 [33, 29] 또는 하드 예제 마이닝 [37, 8, 31]과 같은 기술을 사용하여 해결됩니다.

* ## bootstrapping 
Bootstrapping은 학습 데이터를 라벨링하는 과정에서 발생하는 오류를 최소화하기 위한 방법 중 하나입니다. 일반적으로, 데이터셋에서 일부 샘플은 라벨링이 잘못되어 있거나 누락되어 있을 수 있습니다. 이러한 문제는 기계 학습 알고리즘의 성능을 저하시키고, 모델이 잘못된 결정을 내리게 할 수 있습니다.<br><br>
Bootstrapping은 이러한 문제를 해결하기 위해 사용되는 반복 알고리즘입니다. <br><br>이 방법은 초기 모델을 사용하여 데이터셋을 학습한 후, 해당 모델을 사용하여 예측을 생성합니다. 그런 다음, 예측에서 일부 샘플을 선택하고, 해당 샘플을 라벨링한 다음 학습 데이터에 추가합니다. 이렇게 추가된 데이터를 사용하여 모델을 업데이트하고, 새로운 예측을 생성합니다. 이 프로세스는 반복적으로 수행되며, 모델이 잘못된 예측을 하거나 분류기의 확신이 낮은 예제를 집중적으로 학습하도록 하는 데 도움이 됩니다.<br><br>
Bootstrapping은 다른 데이터 확장 기술과 함께 사용할 수 있으며, 특히 학습 데이터셋이 작거나 불균형 할 때 효과적입니다. <br><br>이 방법은 일반적으로 대규모 데이터셋에서 사용되며, 실제로 자동차, 얼굴, 동물 등 다양한 객체를 인식하는 데 사용되는 딥러닝 모델에서 효과적이라는 것이 입증되어 있습니다

* ## hard example mining 
Hard example mining은 딥러닝 모델에서 사용되는 훈련 기술 중 하나로, 모델이 틀리게 예측한 샘플들 중에서 가장 어려운 샘플들을 선별해 다시 학습시켜 모델의 성능을 향상시키는 방법입니다.<br><br>
딥러닝 모델이 훈련 데이터셋에 대해 학습을 하면, 주어진 입력에 대해 정확한 출력을 예측하도록 가중치를 조정합니다. 하지만, 모든 입력 데이터에 대해 항상 정확한 출력을 예측하는 것은 어렵기 때문에, 모델은 일부 입력 데이터에 대해 틀리게 예측할 수 있습니다.<br><br>
이 때, Hard example mining은 이러한 모델의 틀린 예측을 해결하기 위해 사용됩니다. Hard example mining은 모델이 틀리게 예측한 샘플들 중에서 가장 어려운 샘플들을 선별해 다시 학습시킵니다. 이렇게 하면 모델이 어려운 샘플들에 대해 더 잘 예측하도록 학습되어 전체적인 성능을 향상시킬 수 있습니다.<br><br>
Hard example mining은 대부분의 객체 감지 모델에서 사용되며, 일반적으로 훈련 중에 일정 비율로 가장 어려운 샘플들을 선택해 사용합니다. 이를 통해 모델의 성능을 향상시킬 수 있습니다.

이 논문에서는, 이전의 클래스 불균형 처리 방법에 대한 보다 효과적인 대안으로 작용하는 새로운 손실 함수를 제안합니다. <br><br>이 손실 함수는 동적으로 조정되는 크로스 엔트로피 손실입니다. 이 손실 함수는 동적으로 스케일링된 교차 엔트로피 손실이며 올바른 클래스에 대한 신뢰도가 증가함에 따라 스케일링 계수가 0으로 감소합니다(그림 1 참조). <br><br> 직관적으로, 이 스케일링 요소는 학습 중 쉬운 example들의 기여도를 자동으로 낮추고 모델을 빠르게 어려운 examples에 집중하도록 할 수 있습니다. 실험 결과, 제안된 Focal Loss를 사용하면 샘플링 휴리스틱 또는 하드 예제 마이닝으로 학습하는 것보다 더 높은 정확도를 가진 원스테이지 검출기를 훈련시킬 수 있습니다. 이전 원스테이지 검출기 훈련 기술의 최첨단 기술입니다. <br><br> 마지막으로, 포컬 손실의 정확한 형태는 중요하지 않다는 것을 언급하며, 다른 구성 요소가 유사한 결과를 달성할 수 있다는 것을 보여줍니다.

제안된 포컬 로스의 효과를 입증하기 위해, 우리는 RetinaNet이라는 간단한 원 스테이지 객체 탐지기를 설계했습니다. 이는 입력 이미지에서 객체 위치를 밀도있게 샘플링하는 것을 특징으로 하는데, 효율적인 인네트워크 피라미드와 앵커 박스 사용을 결합한 디자인입니다. <br><br>이는 최근의 [22, 6, 28, 20] 아이디어를 활용합니다. RetinaNet은 효율적이면서도 정확하며, ResNet-101-FPN 백본을 기반으로 한 최상의 모델은 5 fps에서 실행되면서 COCO test-dev AP 39.1을 달성하며, 이전에 출판된 단일 및 이중 스테이지 검출기의 최상의 결과를 능가합니다. (그림 2 참조)

# 2. Related Work

클래스 불균형(Class Imbalance) : 고전적인 원 스테이지 객체 검출 방법인 boosted detectors [37, 5] 및 DPMs [8]와 최근의 방법인 SSD [22] 모두 훈련 중 큰 클래스 불균형에 직면합니다. 이러한 검출기는 이미지 당 104-105개의 후보 위치를 평가하지만 몇몇 위치만이 객체를 포함합니다. <br><br>이러한 불균형은 두 가지 문제를 발생시킵니다: (1) 대부분의 위치가 유용한 학습 신호를 제공하지 않는 쉬운 부정적 예제일 때 훈련이 비효율적이며, (2) 쉬운 부정적 예제가 대량으로 훈련을 압도하고 비정상적인 모델로 이어질 수 있습니다. 일반적인 해결책은 하드 네거티브 마이닝 [33, 37, 8, 31, 22]을 수행하여 훈련 중 어려운 예제를 샘플링하거나 더 복잡한 샘플링/재가중치 방법[2]을 사용하는 것입니다. <br><br> 이에 대비하여, 우리는 제안하는 포컬 로스(Focal Loss)가 원 스테이지 검출기가 직면하는 클래스 불균형을 자연스럽게 처리하며, 샘플링 없이 모든 예제를 효율적으로 훈련할 수 있도록 하고, 쉬운 부정적 예제가 로스와 계산된 기울기를 압도하지 않도록 합니다.

강건 추정(Robust Estimation): Huber loss와 같은 강건한 손실 함수를 설계하는 것에 대한 관심이 많았습니다. 이러한 함수는 오류가 큰 샘플(어려운 샘플)의 손실을 가중치를 낮추어 아웃라이어의 영향력을 줄입니다. <br><br>하지만 우리의 focal loss는 이와 달리, 이너(쉬운 샘플)의 기여를 줄여 클래스 불균형에 대처하도록 설계되었습니다. 이너의 수가 많아도 전체 손실에 대한 기여가 작아집니다. <br><br>즉, focal loss는 강건한 손실 함수의 역할을 수행하는 대신, 적은 수의 어려운 샘플에 학습을 집중하도록 합니다.


```python

```
