# Image-Classification-and-Recommendation-System-on-Walmart-Products

# Project Description

The aim of the project is to classify Walmart products and to make recommendations based on images and titles. The images are based on 13 categories of sports scraped from Walmart website.


# Project Methodology

- **Getting data**: Scraped data from the Walmart website using the BlueCart API and collected product information including product titles, categories, and image URLs.

- **Data Preprocessing**: Preprocessed the image data and filtered out GIF images.

- **Image Classification**: Used Convolutional Neural Networks (CNNs) for image classification.

- **Recommendation system**: Utilized a pre-trained VGG16 model for feature extraction in the recommendation system.

## I. Image Classification component: Convolutional Neural Network (CNN)

### 1. Model Architecture:
- Convolutional Layers: Three convolutional layers are employed. The first layer has 64 filters, followed by max-pooling with a (2, 2) window size and 20% dropout. The subsequent layers have double the number of filters and follow the same pattern.
- Fully Connected Layers: After flattening the output from the convolutional layers, there is a dense layer with 64 neurons activated by ReLU, followed by a 20% dropout.
- Output Layer: The output layer consists of 13 neurons, corresponding to the number of classes in the dataset, with softmax activation.
### 2. Rationale and Modifications:
- Regularization: L1 regularization is applied to the convolutional layer to prevent overfitting by penalizing large weights.
- Dropout: Dropout layers with a dropout rate of 20% are inserted after each max-pooling layer and the fully connected layer to further mitigate overfitting.
- Activation Function: ReLU activation is chosen for all layers except the output layer, where softmax is used to get class probabilities.
### 3. Optimize model performance by tuning hyperparameters:
- Hyperparameters: dense_units, dropout rate, filters, learning rate
- Best combination: dense_units=64, dropout rate=0.2, filters = 64, learning rate= 0.001

### **Results**:

- Model Training Performance:

| Experiment | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss |
| ---------- | -------------- | ------------------- | ---------- | --------------- |
| Base Model | 72.02%         | 52.29%              | 0.8540     | 1.8072          |

```
   (1) The base model achieved a training accuracy of 72.02% and a validation accuracy of 52.29% after 10 epochs
   (2) Early stopping was employed to prevent overfitting
```

- Model Testing Performance:

| Test Accuracy: | 52.52% |
| -------------- | ------ |
| Test Loss:     | 1.7492 |

## II.Text and Image-based Recommendation Component
A combination of image feature extraction using a pre-trained VGG16 model, NLP, and collaborative filtering.
### 1. Image-based recommendation
- Etracted features using a pre-trained VGG16 model
- computed cosine similarities between the feature vectors to identify the similarity between products
### 2. Text-based recommendation 
- Used CountVectorizer from scikit-learn to convert the text data (titles) into numerical vectors
### 3. Combination of image and text recommendation
- Designed the combined_recommend function combines both textual (word-2-vec) and image features to recommend similar products, allowing to specify weights for each feature type.
