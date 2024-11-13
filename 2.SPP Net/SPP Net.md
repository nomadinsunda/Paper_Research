## SPP Net(Spatial Pyramid Pooling Network)


### 정의

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/0fa40e78-5bb3-43e6-af3c-b0cce528fe8a)


RCNN은 기존에 입력 이미지가 고정되어야 했습니다. 이는 입력 이미지의 크기가 일정해야만 신경망을 효과적으로 통과시킬 수 있기 때문입니다. 그러나 이로 인해 이미지를 고정된 크기로 크롭하거나 비율을 조정해야 했습니다. 이는 물체의 일부분이 잘리거나 왜곡되는 문제를 일으킬 수 있었습니다. 또한, RCNN은 각 이미지를 Convolution 연산을 통해 처리하기 때문에 속도가 느려지는 단점이 있었습니다.

이러한 문제를 해결하기 위해 Spatial Pyramid Pooling이 도입되었습니다. 이 방법은 고정된 임의 크기의 입력 이미지를 받아들일 수 있습니다. 이를 통해 이미지 크기나 비율에 대한 제약을 없애고, 더욱 다양한 크기와 비율의 객체를 효과적으로 탐지할 수 있게 되었습니다. 따라서 RCNN에서 Spatial Pyramid Pooling을 사용함으로써 모델의 성능과 효율성을 향상시킬 수 있었습니다.

### **전체 알고리즘 실행 순서**

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/5b45fba8-8b47-4761-982b-95e0db8cacf5)


1. 먼저 전체 이미지를 미리 학습된 CNN을 통과시켜 피쳐맵을 추출한다.

2. Selective Search를 통해서 찾은 각각의 RoI들은 제 각기 크기와 비율이 다르다.

   이에 SPP를 적용하여 고정된 크기의 feature vector를 추출한다.


> (이때 Selective Search는 Input image 에서 한다)

✅ 디테일 설명
*   4 $\times$ 4,  2 $\times$ 2,  1 $\times$ 1 세가지 영역으로 나누고 이것을 하나의 피라미드라고 한다


*  어떤 이미지가 들어 와도 분할영역이 16,4,1로 고정이 되어 있어, 영역의 개수는 21개로 고정이 되어 있다.
*   16, 4, 1로 나눈 값에서 가장 큰 값만 추출하는 max pooling을 수행하고, 그 결과를 쭉 이어붙여 준다
*   입력 피쳐맵의 채널 크기를 k, bin의 개수를 M이라고 할 때 SPP의 최종 아웃풋은 채널과 bin개수의 벡터가 나온다
입력 이미지의 크기와 상관없이 미리 설정한 bin의 개수와 CNN 채널 값으로 SSP의 출력이 결정됨으로 항상 동일한 크기의 결과를 리턴한다.

3. 그 다음 fully connected layer들을 통과 시킨다

4. 앞서 추출한 벡터로 각 이미지 클래스 별로 binary SVM(Support Vector Machines) Classifier를 학습시킨다

5. 마찬가지로 앞서 추출한 벡터로 bounding box regressor를 학습시킨다

✅ Full connective Layer의 한계 극복 방법

 **공간정보의 손실 방지**

Bag of Word는 이미지의 특징을 추출하여 표현하는 기법 중 하나인데, 특정 영역에 대해 이미지의 주요 feature들을 뽑아내는 것입니다. 그러나 이 기법의 단점 중 하나는 지역정보의 손실입니다. 즉, 이미지의 특정 부분에서 발생한 패턴이나 특징이 무시되는 경우가 있습니다.

이러한 문제를 해결하기 위해 이미지를 여러 영역으로 나눈 후 각 영역에 Bag of Word를 적용하는 방법이 있습니다. 예를 들어, 이미지를 n개의 등분으로 나눈 후 각각의 부분에 대해 Bag of Word를 적용할 수 있습니다. 이를 통해 이미지의 지역정보가 어느 정도 보존될 수 있습니다.

이것을 피라미드 모양으로 구성한 것을 Spatial Pyramid라고 합니다. 이 방법은 이미지를 여러 단계로 나누어 각 단계마다 Bag of Word를 적용하는 방식으로, 이미지의 다양한 크기와 해상도에 대해 효과적으로 작동할 수 있습니다. 따라서 Spatial Pyramid를 사용하면 지역정보의 손실을 방지하고, 보다 정확한 이미지 특징 추출이 가능해집니다.





# Bag of Word

# Spatial Pyramid Matching

# Efficient Subwindow Search

### SPP-Net을 Object Detection에 적용

![image](https://github.com/KKH0228/Paper-Research/assets/166888979/0496516f-74de-4abe-818e-2433196d4aa9)


-  
RCNN은 selective search를 사용하여 약 2천개의 물체 영역(Region of Interest, RoI)을 찾은 후, 이를 모두 고정된 크기로 조절한 다음 미리 학습된 CNN을 통과시켜 feature를 추출합니다. 이는 처리과정이 매우 복잡하고 속도가 느려지는 단점을 가지고 있습니다

  
-  SPP Net은 입력 이미지를 그대로 CNN에 통과시켜 feature map을 추출한 다음, 이 feature map에서 selective search를 통해 약 2천개의 물체 영역을 찾습니다. 그 후에는 Spatial Pyramid Pooling(SPP)을 적용하여 고정된 크기의 feature를 얻어내고, 이를 Fully Connected layer와 SVM Classifier에 통과시킵니다. 


### SPP의 한계점



*   SPP Net은 end-to-end 학습 방식이 아닙니다. 즉, 네트워크를 처음부터 끝까지 전체적으로 학습시키는 것이 아니라 여러 단계로 나누어서 학습해야 합니다. 이는 모델의 복잡도를 증가시키고 학습 프로세스를 더 복잡하게 만듭니다.
*   최종 분류는 여전히 이진 SVM을 사용하고, Region Proposal은 여전히 Selective Search를 사용합니다. 이는 전체 시스템을 통합하여 최적화하는 데 어려움을 초래할 수 있습니다.
*   SPP를 거치기 전의 Convolutional Layer들을 fine-tuning하는 것이 어려울 수 있습니다. 일반적으로 SPP를 통과한 후에는 Fully Connected Layer만 학습시키게 됩니다. 이는 네트워크의 일부를 더 적극적으로 최적화하지 못하게 되는데, 이는 모델의 성능을 제한할 수 있습니다.


### 참조 [1](https://yeomko.tistory.com/14)


```python

```
