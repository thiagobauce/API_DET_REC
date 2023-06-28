# Facial Recognition API
This API provides facial recognition services using state-of-the-art models. It includes features such as fine-tuning a pretrained zero-shot model and evaluation using the MegaFace dataset.

## Usage
To train the model, execute the following command:

bash
python train.py configs/megaface

To optimize resource usage and achieve optimal performance, adjust the sample_rate parameter in the configuration file to a value less than 1.0.

For evaluation, the files verification_v3.py and verification_v4.py handle image pairs and utilize cosine similarity from PyTorch for determining positive and negative pairs.

To test the model using sklearn metrics, apply k-nearest neighbors (knn) on a dataset structured with identities in subdirectories. Modify the eval_knn_metrics.py program to specify the desired directory.

The recognize.py script is used by the API to perform facial recognition. It searches a directory containing predefined identities.

## Requirements
Python 3.x
PyTorch
Scikit-learn

## Results
Our zero-shot model got 0.998 in lfw test and 0.913 in cplfw. 

We made a experiments in multiple layers from Resnet50.

1. L4: Training only layer 4 and freezing the others.
2. L3: Training only layer 3 and freezing the others.
3. L34: Training only layers 3 and 4 and freezing the others.
4. L234: Training only layers 2, 3, and 4 and freezing the others.
5. L4FC: Training only layer 4, freezing the others, and modifying the fully-connected layer and the classifier layer.
6. L3FC: Training only layer 3, freezing the others, and modifying the fully-connected layer and the classifier layer.
7. L34FC: Training only layers 3 and 4, freezing the others, and modifying the fully-connected layer and the classifier layer.
8. L234FC: Training only layers 2, 3, and 4, freezing the others, and modifying the fully-connected layer and the classifier layer.
9. L4ZS: Training only layer 4, freezing the others, but trained with the concatenated datasets used in the Mega Face + Family Experiment.

Note: These are different training configurations for the model, where specific layers are trained while the others are frozen. The configurations may also involve modifying the fully-connected layer and the classifier layer and the results were:

L4 0.874
L3 0.725
L34 0.734
L234 0.695
L4FC 0.868
L3FC 0.695
L34FC 0.686
L234FC 0.695
L4ZS 0.996

## References
-[ArcFace Partial FC](https://github.com/deepinsight/insightface)
