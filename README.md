🧠 Overview of the Project
This project utilizes Convolutional Neural Networks (CNNs) to classify plant leaf images into various disease categories. The workflow involves data preprocessing, model training, and disease prediction on new images.​

📁 training.ipynb – Model Training
1. Data Loading and Preprocessing
Dataset: The model is trained on a dataset of plant leaf images, each labeled with the corresponding disease.​

Preprocessing Steps:

Resizing images to a uniform dimension.

Normalizing pixel values.

Splitting the dataset into training and validation sets.​

2. Model Architecture
CNN Structure:

Multiple convolutional layers to extract features.

Pooling layers to reduce dimensionality.

Fully connected layers leading to the output layer.​

Activation Functions: ReLU for hidden layers and Softmax for the output layer.​

3. Training Process
Loss Function: Categorical Crossentropy.​

Optimizer: Adam optimizer for efficient training.​

Metrics: Accuracy to evaluate model performance.​

Epochs: The model is trained over multiple epochs with batch processing.​

4. Model Evaluation and Saving
The trained model is evaluated on the validation set.​

Performance metrics and loss curves are plotted.​

The final model is saved for future inference.​

🧪 PlantDiseasesTest.ipynb – Model Testing
1. Model Loading
The previously saved model is loaded into the environment.​

2. Image Preprocessing
New images are loaded and preprocessed to match the training data format.​

3. Prediction
The model predicts the disease category of the input image.​

The predicted label is displayed alongside the input image
