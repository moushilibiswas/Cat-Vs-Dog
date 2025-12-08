# Cat-Vs-Dog

## 🐶🐱 Dogs vs Cats Classification using Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) to classify images of **dogs** and **cats** using the **Dogs vs Cats** dataset from Kaggle.
The workflow includes dataset download, extraction, preprocessing, CNN model development, training, evaluation, and prediction on test images.

---

## 🚀 **Project Overview**

### **1. Dataset**

* **Source:** Kaggle — Dogs vs Cats
* **URL:** [https://www.kaggle.com/datasets/moazeldsokyx/dogs-vs-cats](https://www.kaggle.com/datasets/moazeldsokyx/dogs-vs-cats)
* The dataset contains **20,000 training images** and **12,461 testing images**, divided into:

  * `/cats`
  * `/dogs`

### **2. Environment Setup**

The Kaggle API key is placed in the appropriate directory:

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
#!chmod 600 ~/.kaggle/kaggle.json   # recommended for security
```

Dataset download:

```bash
!kaggle datasets download moazeldsokyx/dogs-vs-cats
```

Extraction:

```python
import zipfile
with zipfile.ZipFile('dogs-vs-cats.zip', 'r') as zip_ref:
    zip_ref.extractall()
```

---

## 🧹 **3. Data Preprocessing**

Images are loaded via:

```python
train_ds = keras.utils.image_dataset_from_directory(
    '/content/dataset/train',
    labels='inferred',
    batch_size=32,
    image_size=(256, 256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    '/content/dataset/test',
    labels='inferred',
    batch_size=32,
    image_size=(256, 256)
)
```

Normalization layer:

```python
def process(image, label):
    image = tf.cast(image / 255., tf.float32)
    return image, label
```

---

## 🧠 **4. Model Architecture (CNN)**

The model uses:

* 3 × Convolutional layers
* MaxPooling after each Conv layer
* Flatten
* Dense (128 → 64 → 1)
* Sigmoid output for binary classification

Summary:

```
Total Params: 14,847,297
Trainable Params: 14,847,297
```

Model compilation:

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

---

## 🎯 **5. Model Training**

```python
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
```

Training reached:

* **Training accuracy:** 99%
* **Validation accuracy:** ~78–79%
  (indicates mild overfitting)

---

## 📊 **6. Performance Visualization**

Training vs Validation Accuracy:

```python
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
```

Training vs Validation Loss:

```python
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
```

---

## 🖼️ **7. Testing with Custom Images**

Steps:

1. Read image using `cv2.imread`
2. Resize to `(256, 256)`
3. Reshape for batch dimension
4. Predict using `model.predict`

Example:

```python
test_img = cv2.imread('/content/dataset/test/cats/cat (1).jpg')
test_img = cv2.resize(test_img, (256, 256))
test_input = test_img.reshape((1, 256, 256, 3))
model.predict(test_input)
```

Output:

* `[[0.]]` → Cat
* `[[1.]]` → Dog

---

## 📦 **8. Requirements**

* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV (cv2)
* Kaggle API

---

## 🧩 **9. Future Improvements**

* Add Data Augmentation to reduce overfitting
* Use Transfer Learning (VGG16, ResNet50, MobileNet)
* Implement EarlyStopping & Dropout layers
* Hyperparameter tuning
* Save and deploy as a web app (Flask/Streamlit)

---

## 👨‍💻 **10. Author**

**Mistu Biswas**
Deep Learning / Machine Learning Practitioner

