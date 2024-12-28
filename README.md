Human Activity Recognition Using Deep Learning
Overview
This project focuses on Human Activity Recognition (HAR) using deep learning models trained on image data. The aim is to classify various human activities, such as "sitting," "hugging," and "using a laptop," using Convolutional Neural Networks (CNNs). The project demonstrates the use of CNNs for image classification tasks and explores optimization techniques to enhance performance.

preprocessing
Link to dataset : https://dphi-live.s3.eu-west-1.amazonaws.com/dataset/Human+Action+Recognition-20220526T101201Z-001.zip

Loading Dataset and CSV Files: Paths to the training and testing images are specified. The CSV files that contain the labels (Training_set.csv, Testing_set.csv) are read using pandas.
Dataset Exploration: Displaying the first few rows of the training dataset and checking for missing values, duplicates, and dataset statistics. From the output, there are 12,600 records, and no missing or duplicate values.
Visualizing Sample Images: Images from different classes are displayed in a 3x5 grid to give an overview of the dataset. A histogram of the class distribution is generated to show that there is no class imbalance.
Outlier Detection (Z-Score Method): Using the z-score method, the dataset is checked for outliers, and the result shows that there are no rows with outliers in the numerical data.
Image Intensity Distribution: A pixel intensity histogram for grayscale images is plotted, showing that most pixels have lower intensity values, with a significant peak near the maximum intensity (indicating possibly bright areas in the images). Z-scores of pixel intensities are calculated to detect outlier pixels, which result in zero outliers.
Edge Intensity Distribution: The Sobel edge detection filter is applied to the images, and a histogram of mean edge intensities is plotted. The distribution shows that most images have low to moderate edge intensity, indicating relatively smooth regions in the dataset.
Data Augmentation: Data augmentation is applied to the training images using transformations like resizing, random horizontal flips, rotations, and color jittering to improve the diversity of the training data. Validation images are resized but not augmented (no transformation applied).
Custom Dataset Class and DataLoader: A custom PyTorch dataset class (CustomImageDataset) is defined to load the images and their labels, with optional transformations applied to the images. The training and validation sets are split, and DataLoader objects are created for both the training and validation datasets.

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


Model :
Optimizer: SGD with momentum
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy

3. Optimization Techniques
Hyperparameter Tuning: Adjusted learning rates and batch sizes.
Regularization: Used dropout layers and L2 regularization.
Data Augmentation: Rotations, shifts, zooms, and flips to enhance data diversity.

Results
Training Accuracy: Achieved a peak of 98% after 30 epochs.
Validation Accuracy: Reached 81%, demonstrating robust performance across classes.
F1-Score: Exceeded 0.81 for most activities.

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
Copy code
pip install -r requirements.txt

Run the Jupyter Notebook:

Copy code
jupyter notebook har_final.ipynb

References
TensorFlow Documentation. TensorFlow. https://www.tensorflow.org/
Keras API Reference. Keras. https://keras.io/
IEEE Template Guidelines. IEEE. https://www.ieee.org/
