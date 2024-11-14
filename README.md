# Towards-Smart-Waste-Segregation-Leveraging-Deep-Learning-for-Effective-Waste-Classification

# INTRODUCTION :
Waste management is a critical challenge, particularly in
countries like India, where inefficient waste segregation and
recycling, especially of plastics, significantly contribute to environmental
pollution. To tackle this, advanced technological
solutions are essential. Deep learning, specifically Efficient-
Net, provides a promising approach for automating waste
classification with improved accuracy and efficiency.
This project utilizes the Recyclable and Household Waste
Classification Dataset, which includes 15,000 images across 30
categories of waste. By leveraging this dataset and applying
EfficientNet, the goal is to classify waste into different categories,
facilitating better waste segregation at the household
level. The system aims to support efficient resource recovery,
promote recycling, and reduce the environmental impact of
mismanaged waste.

# OBJECTIVE OF THE STUDY :
The main objective of this study is to develop an advanced
waste classification and prediction system using deep learning
techniques. The system will employ EfficientNet to classify
waste into different categories such as plastic bottles, paper
cups, and textiles. By integrating data augmentation methods,
Cross-Entropy Loss for robust multi-class classification and
Grad-CAM for visual interpretability, this research aims to
enhance waste segregation at the household level. Additionally,
a user-friendly web application has been developed, allowing
users to upload images of waste for real-time classification
predictions. The goal is to improve resource recovery, promote
effective recycling, and contribute to reducing environmental
pollution.

# DATASET :
The Recyclable and Household Waste Classification Dataset
is used, which consists of 15,000 images across 30 waste
categories, including plastic, paper, metal, organic, and textile
waste. The dataset is diverse and represents real-world waste
classification challenges. Each image is pre-labeled with its
corresponding category, and preprocessing steps are applied
to ensure uniformity and consistency in the data.

![4](https://github.com/user-attachments/assets/93c7dcef-c5fb-4183-a196-9f0c5d9c2faa)

# DATA PREPROCESSING :
Data preprocessing is crucial to ensure that the model is
trained on clean and standardized data. The following steps
are implemented:
• Resizing: All images are resized to a consistent input size
of 224x224 pixels to match the input requirements of the
EfficientNet model.
• Normalization: Image pixel values are scaled between 0
and 1 to improve model convergence during training.
• Data Augmentation: Techniques such as random rotation,
zoom, horizontal flipping, and shifting are applied
to increase the diversity of the training dataset and reduce
overfitting.

ORIGINAL IMAGE :
![Image_6](https://github.com/user-attachments/assets/287d7808-440f-4eda-b455-2fcfa22baca0)

TRANSFORMED IMAGE :
![7](https://github.com/user-attachments/assets/0830a69d-f2a1-4a11-be2a-4f42c2eaf533)

# MODEL ARCHITECTURE :
The model architecture for this waste classification project
utilizes EfficientNet, a state-of-the-art deep learning model
that is known for its efficiency and performance, especially
in image classification tasks. EfficientNet is chosen for this
project due to its ability to balance accuracy with computational
efficiency, which is crucial for deploying in real-time
waste segregation applications.
EfficientNet uses a compound scaling method to balance
the depth, width, and resolution of the network. This scaling
approach makes it computationally efficient while achieving
excellent performance, which is essential for handling largescale
image datasets like those in waste classification.
The architecture of the model for waste classification is as
follows:
• Input Layer: The input layer accepts preprocessed images
of size 224x224x3. Each image represents a piece
of waste that has been resized and normalized before
being passed into the model. This preprocessing ensures
uniformity and consistency in the input data, allowing
the model to focus on the important features within the
images.
• Convolutional and Pooling Layers: EfficientNet’s backbone
contains a series of convolutional layers that extract
features from the input waste images. These layers help
the model detect essential patterns and textures in the
images, such as the shape and color of different types of
waste (e.g., plastic bottles, paper cups, metal cans). After
each convolution operation, pooling layers (specifically
max-pooling) are applied to reduce the spatial dimensions
of the feature maps, allowing the model to retain the most
critical features and improve computational efficiency.
This feature extraction process enables the model to
identify distinct waste categories based on the visual
characteristics of the images.
• Dense Layers: After the feature extraction, the image
data is passed through fully connected (dense) layers.
These layers learn higher-level representations and capture
complex relationships between the features extracted
by the convolutional layers. The dense layers essentially
help the model understand the broader patterns within
the data that define the various waste categories. For
instance, these layers allow the model to distinguish
between plastic and paper waste based on the learned
features.
• Output Layer: The final output layer is a softmax activation
function that generates the predicted probabilities
for each of the 30 waste categories. The model assigns a
probability to each class (e.g., plastic bottles, paper cups,
metal cans) indicating the likelihood that the input image
belongs to that category. The softmax function ensures
that the sum of the probabilities across all classes equals
1, providing a normalized and interpretable output where
the model’s prediction corresponds to the category with
the highest probability.
In this project, EfficientNet is initially pre-trained on the
ImageNet dataset, providing the model with a solid base
for recognizing fundamental image features such as edges,
textures, and shapes. This pre-training accelerates the convergence
process and enhances the accuracy of the model when
fine-tuned on waste classification images. To tailor the model
for waste classification, a custom classifier is added, aligned
with the 30 specific categories in our dataset, allowing the
model to classify images into one of these waste categories.
This architecture is designed to work efficiently with realworld
waste images and provide accurate predictions that can
help with waste segregation in real-world applications.

# GRAD - CAM FOR INTERPRETABILITY :
In the waste classification system, Grad-CAM (Gradientweighted
Class Activation Mapping) is utilized to enhance
interpretability by providing visual explanations of the model’s
predictions. This technique highlights the specific regions of
an image that are most influential in the model’s decisionmaking
process. By generating heatmaps that focus on the
key features of waste items such as shape, color and texture,
Grad-CAM helps users understand which aspects of the image
contributed to the classification of a particular waste category
(e.g., plastic, paper, metal). This interpretability improves user
trust in the system, enabling more transparent decision-making
in waste segregation applications.

![8](https://github.com/user-attachments/assets/43177b99-c34b-4337-87e2-0a53e85248b9)

# RESULTS AND DISCUSSIONS :
The model was trained for 25 epochs, and the training and
validation performances were evaluated at each epoch. Below
is an analysis of the key results observed during the training
process.
In the early epochs, the model showed significant improvement
in both training and validation performance. In the first
epoch, the training accuracy was 53.74%, while the validation
accuracy was 74.10%. By the 5th epoch, the training accuracy
reached 81.42%, and the validation accuracy was 84.80%.
This demonstrates the model’s effective learning ability in
distinguishing between different categories of waste.
As training progressed, both training and validation losses
continued to decrease, while the accuracy improved steadily.
By epoch 8, the model achieved a training accuracy of 90.58%
and a validation accuracy of 90.73%. This indicates that the
model was able to generalize well to unseen data and was not
overfitting. By the end of the 25 epochs, the model achieved
a final training accuracy of 97.33% and a validation accuracy
of 92.63%, indicating robust performance across both training
and validation sets.

![11](https://github.com/user-attachments/assets/088ff61e-bc9d-4968-b4e7-238c62f15595)

![9](https://github.com/user-attachments/assets/03b7cccf-70c4-46d4-ac9e-74be9a5fa256)

# MODEL DEPLOYMENT :
After training the model, the final model weights are saved
and integrated into a web application built using Streamlit.
This application allows users to easily upload images of
waste items and receive predictions directly on their local or
remote devices. The user interface provides feedback on the
classification results, enabling individuals to make informed
decisions regarding waste segregation. Additionally, the app
is designed to be user-friendly, with a simple and intuitive
layout that facilitates smooth interaction. The model inference
is processed in real-time, ensuring prompt results, and the
system is optimized for efficient performance, even with
larger datasets. This deployment enables the application of the
waste classification model in real-world scenarios, helping to
automate and streamline waste management processes. The
following shows a sample of the user interface, where an
image is uploaded and the predicted class of the waste item
is displayed.

![Screenshot 2024-11-12 101857](https://github.com/user-attachments/assets/c92f9258-3a4c-4e48-bb03-aecac1f9b3d8)

# CONCLUSION :
This study presents a waste classification model utilizing
EfficientNet, achieving a notable test accuracy of 93.57%. This
performance surpasses several existing models in the domain
of image classification for waste management. For comparison,
MobileNet achieved an accuracy of 82%, ResNet50 showed
91%, and ResNet18 achieved 92%. The EfficientNet-based
model demonstrates superior accuracy and robustness, making
it highly effective for real-world waste classification tasks.
The results suggest that EfficientNet’s architecture, with
its ability to efficiently balance model complexity and performance,
is particularly well-suited for waste classification.
This model not only outperforms traditional architectures but
also provides an efficient and scalable solution for waste
management systems, contributing to more sustainable waste
handling practices.
With further improvements, such as the integration of a
more diverse dataset and expansion of the classification categories,
the model can be refined for even greater accuracy.
Additionally, real-time predictions in a user-friendly web application
can assist users in making informed decisions about
waste segregation, thereby promoting better waste management
practices.
