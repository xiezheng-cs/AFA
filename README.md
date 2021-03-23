# Towards Effective Deep Transfer via Attentive Feature Alignment

We provide PyTorch implementation for "[Towards Effective Deep Transfer via Attentive Feature Alignment]"(https://pdf.sciencedirectassets.com/271125/1-s2.0-S0893608021X00032/1-s2.0-S0893608021000307/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEH4aCXVzLWVhc3QtMSJIMEYCIQC3T08gvdE%2BIgOCBefV3EhsR717HwuTW0zrvYOtQg%2BmEwIhANuWiuIHJRAPSG1QLu9Q6DGto2BT3tkBnhUhAfIb%2FH%2FiKr0DCLf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQAxoMMDU5MDAzNTQ2ODY1Igyxtv%2FSpsmlQoCnEakqkQMWHQm9U0ejrtsi6PlxjfUeBlfwV2Xee9mhVrTSOiQ9PUvDBbZYOWzXdzm65WScWKfyPc8fZLZtHa0o2N8GSosrwQIidW%2BapruVhOOZ1w6VCBaUk8XJwl05lzkIW9xR7NOjUIC5t6vNDXtnA%2BwABDlz8EZ%2BD3XJ55lRPtrIoTWFD%2BULHEvFivfMseMcNp8uqtbfvEmItqD%2Buoqq23Uz%2FCDevG5L2iqxYo%2F9bekdys7SkAMHh8Ugrcere9UKRuTbAZ%2Bm9%2By5cIoQqli46O965hm0Iz6XnikreNClH8VmuA8%2BqtS6c7uz6QWStf6jzFvPq6vM7BRKK%2FG6sqOKWoXFD65OUyUW%2FbRsgFMrcH5LdAUB4Vw%2FayDbElsyuyGfjZI2z7Dphw%2BVE0DQLzz1Vv6ggMG2AvjNCFshHERv9LjFbLIIpTQwyc%2Bfhiv4lZLbZqleAD6WPne748swlNuFBCuMB9w5C%2BMfgWBZh0KtBrZwlDun3o4Nl%2F8ee7ujlotXJE3d0XFQ1ZVe%2BDQd%2BWW9k9HBKs7ChDDcheaCBjrqAeFirpNxGmymMqb29dQzRJh0l7Qus1Cd4cG0tSXrD%2B0C9RxPjJYEIfdTtBK8CQTjHRF3uUEGAyupQOitPQmXfe321Z3V5Iapqw1mnoGngBZcSlVyVG6c%2Bg2rJ0%2B88xxY%2BnGo7XXPwO60J%2F8Cas4P%2FlY9oHYk2zt3uyI72Wy8pFWg1W3WXfbmVdcGqW9pGMcQFdO87UilrA%2FS4ZBFQsIKeTgjvuFR4YXn3CaoizT03V6EU%2F48eQPMTh2s9su22HhWMag%2FdWLpMsJ%2BuVV8M6XmRRB3DffgRhv1tBin8ft1JNoaPs2YVIaX6z%2FjKw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210323T071721Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4SL5E572%2F20210323%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=f5a85b7e2b128438f6b07b098ff5b9bfc83a086ae8b883601091af2cff306ed5&hash=8777300117e2539cbd3b587c44735f93d595c878c9e96101e2ac7a6e055b46a2&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0893608021000307&tid=spdf-bad18329-9e57-41bb-9548-c21c4645f997&sid=4866547288373045b74966e-89062388a8fegxrqa&type=client) (NN2021).

<p align="center">
<img src="framework.png" alt="AFA" width="90%" align=center />
</p>


## Dependencies

* Python 3.6
* Pytorch 1.1.0
* Dependencies in requirements.txt

### Installation

1. Clone this repo:

        git clone https://github.com/xiezheng-cs/AFA.git
        cd AFA

2. Install pytorch and other dependencies.

        pip install -r requirements.txt

### Training

1. **First Stage**:
   
        ResNet101: python main.py hocon_config/First_stage_AST/resnet101_stanford_dogs_120.hocon

        MobileNet_V2: python main.py hocon_config/First_stage_AST/mobilenet_v2_stanford_dogs_120.hocon

2. **Second Stage**:

        ResNet101: python main.py hocon_config/Second_stage_ACT/resnet101_Stanford_dogs_120.hocon

        MobileNet_V2: python main.py hocon_config/Second_stage_ACT/mobilenet_v2_Stanford_dogs_120.hocon


## Results

  |  Target Data Set | Model | DELTA Top1 Acc(%) | AFA(Ours) Top1 Acc(%) |
   | :-: | :-: | :-: | :-: |
  | Stanford Dogs 120 | MobileNetV2 | 81.3±0.1 | **82.1±0.1** |
  | Stanford Dogs 120 |  ResNet-101  | 88.7±0.1 | **90.1±0.0** |

<br/>

## Pre-trained Model


 | Model | Link| Top1 Acc (%)|
   | :-: | :-: | :-: |
 |ResNet101|https://github.com/xiezheng-cs/AFA/releases/tag/models| 90.22|
 |MobileNetV2|https://github.com/xiezheng-cs/AFA/releases/tag/models| 82.17|


## Eval
        
        ResNet101: python main.py hocon_config/val/resnet101_Stanford_dogs_120.hocon

        MobileNet_V2: python main.py hocon_config/val/mobilenet_v2_Stanford_dogs_120.hocon

## Citation
If this work is useful for your research, please cite our paper:

    @InProceedings{xie2021afa,
    title = {Towards Effective Deep Transfer via Attentive Feature Alignment},
    author = {Zheng Xie, Zhiquan Wen, Yaowei Wang, Qingyao Wu, and Mingkui Tan},
    journal = {Neural Networks},
    volume = {138},
    pages = {98-109},
    year = {2021}
    }
