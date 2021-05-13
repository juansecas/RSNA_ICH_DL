# RSNA automatic detection of ICH using Deep Learning Framework

### Abstract
Intracranial hemorrhage (ICH) corresponds to any type of bleeding within the skull which is classified as a medical emergency because most deaths occur within the first 24 hours. This is why methods that decrease detection time are important for effective treatment.
In this work, a deep learning model is presented for the multiclass classification of 5 types of intracranial hemorrhage (Epidural, Intraparenchymal, Intraventricular, Subarachnoid, Subdural) using non-contrast computer tomography images. The model makes use of image preprocessing techniques, overtraining reduction, data augmentation, and custom loss function. The model is tested with a database of more than 25,000 non-contrast computed tomography CT imaging studies labeled by expert radiologists in 6 classes.
The results obtained reached a 92.1\% sensitivity in the detection of any type of intracranial hemorrhage. ROC and metric curves are presented to evaluate the performance of the proposed model. A comparison is also made with other similar models found in the literature. The model presented is an important advance in diagnostic support algorithms based on deep learning, which in future works can be used as a support tool in the diagnosis of diseases that use medical images.

In this repository you will find the deep learning propose for [RSNA2019 Intracranial Hemorrhage Detection Challenge](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/overview)

### Model

![Model_V5_Deep](https://user-images.githubusercontent.com/41757003/118171002-c26b3800-b3f8-11eb-937e-4458383c1430.png)


### Data

The model will trained with stage 1 of challege, you can download and retrain the model using [Stage 2 traing data](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data)

### Results

![ROC_Cross-1 (1)](https://user-images.githubusercontent.com/41757003/118172510-75886100-b3fa-11eb-8e14-89f34cffc446.png)

![table_results](https://user-images.githubusercontent.com/41757003/118172788-c13b0a80-b3fa-11eb-9e0d-744ce8c0143a.PNG)

### Model Build
The are two produce the solution:

1. Use `Custom_Model_V5` to train (Change directory of DICOM images)
2. Use `Custom_Model_V6_Cross` to train the model using crossvalidation


Take about 2 per epoch using GPU nVidia 1080Ti
