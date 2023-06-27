## Introduction
This is a University's project to use ArcFace for Facial Recognition. \
In this work we used RetinaFace (Pytorch) with Resnet50 as backbone to detection method and Arcface-PartialFC(Pytorch) with Resnet50 too for Recognition.

## Detection
As we said before, we used RetinaFace (Pytorch).\
Here we trained in GPU Tesla \

Config:\
-8 bacth size \
-Resnet50 as network\
-WiderFace as dataset\
-Multibox Loss as loss\
-100 Epoch with saving each epoch as a checkpoint \

If you need or want, can edit some configs on ./data/config.py\

Train with: \
``python train.py --network resnet50``

Evaluate with: \
``python test_widerface.py --trained_model "model_file.pth" --network resnet50``\
``cd ./widerface_evaluate ``\
``python setup.py build_ext --inplace``\
``python evaluation.py``\

Test with: \
``python detect.py``\


## Results
Comparing to official results we need a good parameter with our training.

+==================== Results ===================+\
    Easy   Val AP: 0.9443050276341179\
    Medium Val AP: 0.9325524894244597\
    Hard   Val AP: 0.8430304248391535\
+================================================+

## References
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Multibox_loss](https://github.com/amdegroot/ssd.pytorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
- [Retinaface (torch)](https://github.com/biubug6/Pytorch_Retinaface)
- [Evaluate Widerface](https://github.com/wondervictor/WiderFace-Evaluation)

