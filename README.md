# Ovarian Cyst Classification

Objective: This project aims to classify ovarian cysts as benign or malignant using deep learning models, leveraging transfer learning in convolutional neural networks (CNNs).

Both the TensorFlow and PyTorch frameworks were used to train on the same dataset with similar model architectures. These two projects are included in this repository to:

- Compare the model performances across both frameworks.
- Ensure that the model is effectively capturing patterns in the medical images.

## Data

The original dataset was sourced from medical journals and provided by doctors. It contained a mix of `.jpg` and `.pdf` files. To ensure the quality and relevance of the dataset, a preprocessing step was performed to eliminate certain files based on the following criteria:

1. Non-image files (e.g., PDFs, text documents)
2. Images with multiple photos in a single file (e.g., collages, comparison images)
3. Annotated images (i.e., images with bounding boxes, arrows, or labels)
4. Images containing text, words, or alphabets
5. Other irrelevant or noisy files that could interfere with model training

Despite these filtering steps, some annotated images were retained to avoid reducing the dataset size too much. After cleaning, the dataset contained:

- Benign images: Reduced from 168 to 148
- Malignant images: Reduced from 103 to 94

The images were structured into a folder format to facilitate preprocessing and model training:

- e2e/
  - Benign/
    - `benign_image1.jpg`
    - `benign_image2.jpg`
    - ...
  - Malignant/
    - `malignant_image1.jpg`
    - `malignant_image2.jpg`
    - ...

The entire dataset folder was zipped and uploaded to Google Drive, which is mounted to Google Colab for training.

## Tensorflow Implementation

[Click here to view the file](./TensorFlow_Ovarian_Cyst_Classification.ipynb)

### Data Preprocessing

- Loading images from respective directories
- Assigning labels: Benign images labeled as [1,0], and malignant images labeled as [0,1]
- Resizing images to 224x224 pixels to maintain uniformity
- Normalizing pixel values to the range 0-1 (originally 0-255)
- Converting images and labels into NumPy arrays for efficient processing
- Splitting data into 80% training and 20% validation sets
- Computing class weights to address class imbalance

### Image Augmentation

- Shifting width by 10% (translation)
- Shifting height by 10% (translation)
- Zoom in/out by 20%
- Flip horizontally
- Only 1 augmentation selected from above for each augmented image
- Fill points in case of outside boundaries with nearest point color

### Model Architecture

Sequential Model:

1. A freezed pretrained DenseNet121 as feature extractor, top layer excluded for transfer learning
2. A global average pooling layer for reducing dimensionality
3. A fully connected layer with 256 neurons and ReLU activation
4. A 0.3 dropout layer for avoiding overfitting
5. An output layer with 2 neurons and softmax activation, ensuring the sum of probabilities for benign and malignant classifications equals 1.

Training Configurations:

- Optimizer: Adam with a learning rate of 0.001 (default)
- Loss Function: Categorical Cross-Entropy
- Callbacks:
  - Early Stopping: stops model training by monitoring validation AUC for 5 epochs
  - Reduce LR on Plateau: multiplies learning rate by 0.2 when validation loss does not decrease for 3 epochs, with a minimum LR of 1e-6
- Training Duration: 50 epochs

### Finetuning

- Fine-tuning is performed only if the baseline model achieves a validation AUC of at least 0.75.
- The last 50 layers of DenseNet121 are unfrozen and trained for 20 additional epochs with a lower learning rate (1e-5).
- Early stopping halts fine-tuning if validation AUC does not improve for 3 consecutive epochs.

### Final Model

The trained models are evaluated using the following metrics:

1. Training vs Validation Accuracy
2. Training vs Validation Loss
3. Training vs Validation AUC

Performance comparisons are visualized through plots, and the best-performing model is selected and saved as a `.h5` file.

To load the model for inference, use:

```python
from tensorflow.keras.models import load_model

model = load_model("model.h5")
prediction = model.predict(img)
benign_prob, malignant_prob = prediction[0]
```

## PyTorch Implementation

[Click here to view the file](./PyTorch_Ovarian_Cyst_Classification.ipynb)

This implementation aims to replicate the findings of a research study, as the original work was not open-sourced. The configurations and parameters used in this project are carefully tuned to align with the details provided in the paper:

[Deep learning-enabled pelvic ultrasound images for accurate diagnosis of ovarian cancer in China: a retrospective, multicentre, diagnostic study](<https://www.thelancet.com/journals/landig/article/PIIS2589-7500(21)00278-8/fulltext>)

### Data Preprocessing

- Loading images from respective directories
- Assigning labels: Benign images labeled as 1, Malignant images labeled as 0
- Resizing images to 224x224 pixels
- Normalizing pixel values to the range 0-1
- Converting images and labels into PyTorch tensors
- Splitting data into 80% training and 20% validation sets
- Computing class weights to address class imbalance

### Image Augmentation

- Random horizontal flip
- Random rotation (up to 30 degrees)
- Random affine transformations (translation and scaling)

### Model Architecture

Sequential Model:

1. A freezed pretrained DenseNet121 as feature extractor, top layer excluded for transfer learning
2. A fully connected layer with 1024 input features and 1 output neuron for binary classification replaced the classifier head

Training Configurations:

- Optimizer: Stochastic Gradient Descent with 0.001 learning rate, 0.9 momentum, 1e-4 weight decay
- Loss Function: Binary Cross-Entropy with Logits Loss
- StepLR Scheduler reduces learning rate by factor of 0.1 every 20 epochs.
- Training Duration: 30 epochs

## References

- [Introduction to DenseNets (Dense CNN)](https://www.analyticsvidhya.com/blog/2022/03/introduction-to-densenets-dense-cnn/)
- [Convolution Neural Network – Better Understanding](https://www.analyticsvidhya.com/blog/2021/07/convolution-neural-network-better-understanding/)
- [Image Classification Using CNN](https://www.analyticsvidhya.com/blog/2020/02/learn-image-classification-cnn-convolutional-neural-networks-3-datasets/)
- [20 Questions to Test your Skills on CNN (Convolutional Neural Networks)](https://www.analyticsvidhya.com/blog/2021/05/20-questions-to-test-your-skills-on-cnn-convolutional-neural-networks/)
- [Deep learning-enabled pelvic ultrasound images for accurate diagnosis of ovarian cancer in China: a retrospective, multicentre, diagnostic study](<https://www.thelancet.com/journals/landig/article/PIIS2589-7500(21)00278-8/fulltext>)
- [Transfer Learning Guide: A Practical Tutorial With Examples for Images and Text in Keras](https://neptune.ai/blog/transfer-learning-guide-examples-for-images-and-text-in-keras)
