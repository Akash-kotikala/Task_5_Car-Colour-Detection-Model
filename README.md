# Task_5_Car-Colour-Detection-Model
Create ML model to identify car colors in traffic, count cars at signals. Draw red rectangles on blue cars, blue on others. Also count people present. Include GUI with input image previews. Prioritize model accuracy and GUI usability.


### Car Model Link ::


https://drive.google.com/file/d/1zXJA4-AIdeuvbLUI77tHzoVU-UyEe3e0/view?usp=drive_link



### Dataset Link ::


https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset



# Car Color Classifier (Blue vs. Other)

## Problem Statement
Classify car images by color, focusing on blue vs. all other colors. This addresses challenges in vehicle recognition systems (e.g., surveillance, autonomous driving) where color imbalance (fewer blue cars) affects accuracy.

## Dataset
- **Source**: Vehicle Color Recognition Dataset (10,373 images across colors like blue, red, black).
- **Preprocessing**: Resized to 64x64, normalized. Binary labels: blue (0), other (1).
- **Classes**: Binary (blue: 1,060 images; other: 9,313).
- **Download**: [Kaggle Vehicle Color Dataset](https://www.kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset).
- **Size**: ~574MB.

## Methodology
1. **Data Loading & Preprocessing**: Load images via OpenCV, resize/normalize. Handle class imbalance implicitly via stratified split.
2. **Model**: Custom CNN with 3 conv layers + pooling, dropout for regularization.
   - Input: 64x64x3 images.
   - Output: Softmax for 2 classes.
   - Optimizer: Adam (lr=0.001).
   - Loss: Categorical Crossentropy.
   - Metrics: Accuracy.
3. **Training**: 80/20 train-test split, 15 epochs. Batch size: 32.
4. **Evaluation**: Accuracy on test set.
5. **Tools**: TensorFlow/Keras, OpenCV, Pandas, Scikit-learn.

## Results
- **Accuracy**: ~99% on test set (from notebook: val_acc ~0.99 after 15 epochs).
- **Precision/Recall**: Near-perfect due to simple binary task, but blue class recall ~0.98.
- **Sample Output**: Model predicts color on new images.
- **Limitations**: Overfits on dominant "other" class; could use augmentation/SMOTE for better balance.

## Installation
```bash
pip install tensorflow opencv-python pandas scikit-learn
