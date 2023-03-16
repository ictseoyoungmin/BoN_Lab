# BoN_Lab

기존 이진 분류나 다중 분류를 수행하는 모델은 훈련되지 않은 클래스의 이미지를 입력하면, 모델의 예측은 훈련된 클래스 중 하나를 예측했다.

OC-CNN의 간단한 확장으로 위 상황에서 훈련에 참여하지 않은 클래스는 Unknown Class로 예측 가능하게 할 수 있다.

예를 들어, Cat과 Dog 클래스를 학습한 이 네트워크는 사자나 표범의 이미지가 입력되면 Unknown으로 예측할 수 있다.

첫 실험이 2가지 클래스와 Unknown Class를 분류하는 것 이었고 마땅히 이 Task에 붙여진 이름이 없는 것 같아 BoN ( Both or Nothing ) Classification 이라 썼다.

