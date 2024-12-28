
Human Activity Recognition Using Deep Learning
Overview
This project focuses on Human Activity Recognition (HAR) using deep learning models trained on image data. The aim is to classify various human activities, such as "sitting," "hugging," and "using a laptop," using Convolutional Neural Networks (CNNs). The project demonstrates the use of CNNs for image classification tasks and explores optimization techniques to enhance performance.

Key Features
Implementation of CNN-based deep learning models for HAR.
Training with Adam and SGD with momentum optimizers.
Comprehensive model evaluation using metrics like accuracy, precision, recall, and F1-score.
Mitigation of overfitting through dropout layers and L2 regularization.
Use of data augmentation to increase dataset variability and robustness.
Dataset
The dataset contains 12,600 labeled images of human activities, with two features: filename and label. It is clean with no missing values. Labels include activities like:

Sitting
Hugging
Using a laptop
Methodology
1. Model Architecture
Convolutional Neural Network (CNN):
Convolutional Layers
MaxPooling Layers
Flattening Layers
Fully Connected Dense Layers
Activation Functions: ReLU and softmax
2. Model Training
Two models were trained:

Model 1:
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy
Model 2:
Optimizer: SGD with momentum
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy
3. Optimization Techniques
Hyperparameter Tuning: Adjusted learning rates and batch sizes.
Regularization: Used dropout layers and L2 regularization.
Data Augmentation: Rotations, shifts, zooms, and flips to enhance data diversity.
Results
Training Accuracy: Achieved a peak of 93% after 20 epochs.
Validation Accuracy: Reached 73%, demonstrating robust performance across classes.
F1-Score: Exceeded 0.70 for most activities.
Challenges and Solutions
1. Overfitting
Challenge: Model initially overfitted the training data.
Solution: Added dropout layers and applied L2 regularization.
2. Computational Limitations
Challenge: Training deep networks required significant computational resources.
Solution: Optimized the training pipeline and utilized GPU acceleration.
Lessons Learned
Preprocessing and augmentation significantly enhance model performance.
Hyperparameter tuning is critical for achieving optimal results.
Visualization tools like confusion matrices provide valuable insights into classification errors.
Future Work
Explore transfer learning with architectures like ResNet or Inception.
Use larger datasets to improve generalization.
Experiment with multimodal data (e.g., video sequences) for HAR.
How to Run the Project
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/your-repo-name.git
Install the dependencies:
bash
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook:
bash
Copy code
jupyter notebook har_final.ipynb
References
TensorFlow Documentation. TensorFlow. https://www.tensorflow.org/
Keras API Reference. Keras. https://keras.io/
IEEE Template Guidelines. IEEE. https://www.ieee.org/
