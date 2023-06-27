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

## References
-[ArcFace Partial FC](https://github.com/deepinsight/insightface)
