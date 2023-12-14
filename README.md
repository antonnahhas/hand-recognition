# hand-recognition

## Image Processing and Classification Pipeline

This Python script provides a comprehensive image processing and classification pipeline. The pipeline consists of several stages, including image preprocessing, model training using transfer learning with a pre-trained ResNet101 model, and evaluation on a test dataset. The script utilizes popular libraries such as OpenCV, NumPy, Seaborn, TensorFlow, and Keras.

### Preprocessing

1. **Loading Data:**
   - The `load_data` function loads peak and image data necessary for preprocessing.

2. **Cleaning Image:**
   - The `clean_img` function cleans the input image by removing certain noise/artifacts.

3. **Connected Black Cells:**
   - The `find_connected_black_cells` function identifies connected black cells in a given grid/image using depth-first search (DFS).

4. **Check Empty Image:**
   - The `checkEmpty` function checks if an image has minimal content or is mostly empty.

5. **Image Processing:**
   - The `process_image` function segments the image into train, validation, and test sets based on given parameters.

6. **Segment Words:**
   - The `segment_words_v12` function segments words from given lines using a minimum word width.

7. **Divide Words:**
   - The `div_words_train_valid` and `div_words_test` functions divide words into train and validation datasets and create word pairs for testing.

8. **Expand and Normalize Image:**
   - The `expand` function expands the image dimensions and normalizes it.

9. **Preprocess and Save Images:**
   - The `preprocess_and_save` function preprocesses and saves the images after segmenting them into train, validation, and test datasets.

### Model Training

1. **Initialization:**
   - Hyperparameters such as batch size, number of epochs, and learning rate are defined.

2. **Data Augmentation:**
   - Image data generators are initialized with preprocessing functions.

3. **Loading Training and Validation Data:**
   - Training and validation data are loaded from preprocessed directories.

4. **Model Architecture:**
   - The script uses transfer learning with a pre-trained ResNet101 model, adding custom layers on top.

5. **Model Compilation:**
   - The model is compiled using the Adam optimizer and categorical crossentropy loss.

6. **Callbacks:**
   - Early stopping and learning rate reduction are implemented as callbacks during training.

7. **Training the Model:**
   - The model is trained using the training data generator.

### Test Evaluation

1. **Loading Test Data:**
   - Test data is loaded from the preprocessed test directory.

2. **Model Evaluation:**
   - The trained model is evaluated on the test dataset.

### Graph Plotting

1. **Training History Plotting:**
   - The `plot_training_history` function visualizes training and validation accuracy as well as training and validation loss.

### Visualization and Interpretation

1. **First Convolutional Layer Filter Visualization:**
   - The script visualizes the filters of the first convolutional layer in the base model.

2. **Confusion Matrix Plotting:**
   - The `plot_confusion_matrix` function plots the normalized confusion matrix based on test predictions.

3. **Error Interpretation:**
   - The script provides error interpretation by analyzing misclassified and correctly classified instances for each true label.

### Note:

- Ensure you have the necessary libraries installed (`cv2`, `os`, `random`, `numpy`, `seaborn`, `scikit-learn`, `tensorflow`, `scipy`, `matplotlib`, `keras`).
- Adjust file paths and directories as needed for your data.