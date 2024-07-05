#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import tensorflow as tf
import keras
import random
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten ,Dropout ,Input , BatchNormalization ,GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score , ConfusionMatrixDisplay
from tensorflow.keras.metrics import Precision , Recall
from keras.metrics import Precision, Recall
import struct
from array import array
from os.path  import join
from keras.models import load_model
from skimage.exposure import rescale_intensity
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau ,LearningRateScheduler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image


# In[2]:


from keras.datasets import cifar100


# In[3]:


(x_train, y_train), (x_test, y_test) = cifar100.load_data()


# In[4]:


np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)


# In[5]:


print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")


# In[6]:


def preprocess_data(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=100)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=100)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train_encoded))

train_dataset = train_dataset.map(preprocess_data)

batch_size = 64
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

for batch in train_dataset.take(1):
    images, labels = batch
    print(images.shape, labels.shape)


# In[7]:


y_train_encoded = to_categorical(y_train, num_classes=100)
y_test_encoded = to_categorical(y_test, num_classes=100)


# In[34]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
import tensorflow as tf
from PIL import Image
import cv2

# تحميل بيانات CIFAR-100
(x_train, _), _ = cifar100.load_data()

# اختيار بعض الصور العشوائية
num_images = 5
random_indices = np.random.choice(len(x_train), num_images)
sample_images = x_train[random_indices]

# تحويل الصور من Tensor إلى NumPy إذا لزم الأمر
sample_images_np = [img if isinstance(img, np.ndarray) else img.numpy() for img in sample_images]

# تحويل الصور إلى نوع uint8
sample_images_np = [img.astype(np.uint8) for img in sample_images_np]

# تحسين دقة الصورة باستخدام PIL
def upscale_image(image, scale_factor):
    img = Image.fromarray(image)
    new_size = (img.width * scale_factor, img.height * scale_factor)
    img_upscaled = img.resize(new_size, Image.BICUBIC)  # استخدام تقنية الاستيفاء البعدي
    return np.array(img_upscaled)

# تطبيق فلتر حاد على الصورة
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    return sharpened

# عرض الصور الأصلية، المكبرة والمحسنة بالفلتر الحاد
plt.figure(figsize=(20, 10), dpi=100)
for i in range(num_images):
    # عرض الصورة الأصلية
    plt.subplot(3, num_images, i + 1)
    plt.imshow(sample_images_np[i])
    plt.title(f"Original Image {i+1}", fontsize=16)
    plt.axis('off')

    # عرض الصورة المكبرة
    img_upscaled = upscale_image(sample_images_np[i], 4)
    plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(img_upscaled)
    plt.title(f"Upscaled Image {i+1}", fontsize=16)
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[37]:



import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler


# تحويل التسميات إلى تصنيف فئة
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=100)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=100)

# إعدادات تعزيز البيانات
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.1,
    fill_mode='nearest'  # استخدام طريقة الملء للحفاظ على جودة الصور
)

# ملاءمة بيانات التدريب على المعزز
datagen.fit(x_train)

# وظيفة لتقليل معدل التعلم كل 10 حلقات
def scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch != 0:
        lr = lr / 2
    return lr

# إعداد الإيقاف المبكر وخفض معدل التعلم
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# بناء النموذج المحسن باستخدام Input
model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.4),

    Conv2D(512, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(100, activation='softmax')
])

# تجميع النموذج مع استخدام Adam
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# تدريب النموذج
history = model.fit(
    datagen.flow(x_train, y_train_encoded, batch_size=64),
    epochs=50,
    validation_data=(x_test, y_test_encoded),
    verbose=1,
    callbacks=[LearningRateScheduler(scheduler), early_stopping, reduce_lr]
)

# حفظ النموذج المدرب
model.save('original_model.h5')


# In[38]:


model = load_model('original_model.h5')

# تقييم النموذج على بيانات الاختبار
loss, accuracy, precision, recall = model.evaluate(x_test, y_test_encoded, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Precision: {precision * 100:.2f}%")
print(f"Test Recall: {recall * 100:.2f}%")
print(f"Test Loss: {loss * 100:.4f}%")


# In[39]:


model = load_model('original_model.h5')

# اختيار بعض الصور العشوائية من مجموعة الاختبار
num_images = 5
random_indices = np.random.choice(len(x_test), num_images)
sample_images_test = x_test[random_indices]

# تحويل الصور من Tensor إلى NumPy
sample_images_test_np = [img.numpy() if isinstance(img, tf.Tensor) else img for img in sample_images_test]

# تحويل الصور إلى نوع uint8
sample_images_test_np = [img.astype(np.uint8) for img in sample_images_test_np]

# تحسين دقة الصورة باستخدام PIL
from PIL import Image

def upscale_image(image, scale_factor):
    img = Image.fromarray(image)
    new_size = (img.width * scale_factor, img.height * scale_factor)
    img_upscaled = img.resize(new_size, Image.BICUBIC)  # استخدام تقنية الاستيفاء البعدي
    return np.array(img_upscaled)

# عرض الصور الأصلية والمحسنة من مجموعة الاختبار
plt.figure(figsize=(20, 10), dpi=100)
for i in range(num_images):
    plt.subplot(2, num_images, i + 1)
    plt.imshow(sample_images_test_np[i])
    plt.title(f"Original Test Image {i+1}", fontsize=16)
    plt.axis('off')
    plt.subplot(2, num_images, i + 1 + num_images)
    img_upscaled = upscale_image(sample_images_test_np[i], 4)
    plt.imshow(img_upscaled)
    plt.title(f"Upscaled Test Image {i+1}", fontsize=16)
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[40]:


model = load_model('original_model.h5')

# رسم منحنيات التدريب
plt.figure(figsize=(18, 10))

# Accuracy
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Loss
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Precision
plt.subplot(2, 2, 3)
plt.plot(history.history['precision'], label='Precision')
plt.plot(history.history['val_precision'], label='Val Precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.title('Training and Validation Precision')

# Recall
plt.subplot(2, 2, 4)
plt.plot(history.history['recall'], label='Recall')
plt.plot(history.history['val_recall'], label='Val Recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.title('Training and Validation Recall')

plt.tight_layout()
plt.show()


# In[41]:


model = load_model('original_model.h5')

model.summary()


# In[44]:


# تحميل بيانات CIFAR-100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# تحويل التسميات إلى ترميز الفئات الثنائية
num_classes = 100
y_train_encoded = to_categorical(y_train, num_classes=num_classes)
y_test_encoded = to_categorical(y_test, num_classes=num_classes)

# دالة لتحويلات الملمس
def apply_texture_transformations(image):
    blurred_image = gaussian_filter(image, sigma=0.5)
    laplacian_image = laplace(blurred_image, mode='reflect') / 4.0
    noise = np.random.normal(0, 0.01, image.shape) * 255
    noisy_image = image + noise
    transformed_image = 0.8 * image + 0.1 * laplacian_image + 0.1 * noisy_image
    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
    return transformed_image

# نسبة التسميم
poison_fraction = 0.5
num_poisoned = int(poison_fraction * len(x_train))
poisoned_indices = np.arange(len(x_train))
x_poison_part = x_train[poisoned_indices]
y_poison_encoded_part = y_train_encoded[poisoned_indices]
x_poisoned = np.array([apply_texture_transformations(img) for img in x_poison_part])
x_train_combined = x_poisoned
y_train_encoded_combined = y_poison_encoded_part

# إعداد مولد بيانات التعزيز
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.1
)

datagen.fit(x_train_combined)

# إعداد الإيقاف المبكر وتقليل معدل التعلم
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# تحميل النموذج الأصلي
model = load_model('original_model.h5')

# إعادة تجميع النموذج الأصلي مع البيانات المسمومة
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy', 'precision', 'recall']  # تم إزالة 'loss' من المقاييس
)

# تدريب النموذج على البيانات المسمومة
history = model.fit(
    datagen.flow(x_train_combined, y_train_encoded_combined, batch_size=64),
    epochs=20,
    validation_data=(x_test, y_test_encoded),
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

# حفظ النموذج باستخدام الدالة المعرفة
model.save('texture_transformed_model.h5')


# In[47]:


model = load_model('texture_transformed_model.h5')

# تقييم النموذج على بيانات الاختبار
loss, accuracy, precision, recall = model.evaluate(x_test, y_test_encoded, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Precision: {precision * 100:.2f}%")
print(f"Test Recall: {recall * 100:.2f}%")
print(f"Test Loss: {loss * 100:.4f}%")


# In[51]:


model = load_model('texture_transformed_model.h5')

# تقييم النموذج لبناء المقاييس
initial_evaluation = model.evaluate(x_test, y_test_encoded, verbose=1)
print(f"Initial evaluation - Loss: {initial_evaluation[0]}, Accuracy: {initial_evaluation[1]}")


# In[62]:



# تحميل النموذج المحول
model = load_model('texture_transformed_model.h5')

# تحديد عدد الصور للعرض
num_samples = 6
random_indices = np.random.choice(len(x_train), num_samples, replace=False)

plt.figure(figsize=(15, 6))
for i, idx in enumerate(random_indices):
    # عرض الصور الأصلية
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(x_train[idx].astype('uint8'))
    plt.title(f'Original {y_train[idx][0]}')
    plt.axis('off')
    
    # عرض الصور المسممة
    plt.subplot(2, num_samples, i + 1 + num_samples)
    plt.imshow(x_poisoned[idx].astype('uint8'))
    plt.title(f'Poisoned {y_train[idx][0]}')
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[48]:


model = load_model('texture_transformed_model.h5')

# عرض الصور الأصلية والمسممة للمقارنة
num_samples = 6
random_indices = np.random.choice(len(x_train), num_samples, replace=False)

plt.figure(figsize=(15, 6))
for i, idx in enumerate(random_indices):
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(x_train[idx].astype('uint8'))
    plt.title(f'Original {y_train[idx][0]}')
    plt.axis('off')

    plt.subplot(2, num_samples, i + 1 + num_samples)
    plt.imshow(x_poisoned[idx].astype('uint8'))
    plt.title(f'Poisoned {y_train[idx][0]}')
    plt.axis('off')
plt.show()


# In[49]:


model = load_model('texture_transformed_model.h5')

plt.tight_layout()
plt.show()

# عرض النتائج من التدريب
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Precision over epochs')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Recall over epochs')
plt.legend()

plt.tight_layout()
plt.show()


# In[64]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import gaussian_filter, laplace
import tensorflow as tf

# Load the trained model
model = load_model('texture_transformed_model.h5')

# Ensure the model is compiled with metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Function for texture transformations with reduced effects
def apply_texture_transformations(image):
    blurred_image = gaussian_filter(image, sigma=0.05)  # Reduce sigma to minimum
    laplacian_image = laplace(blurred_image, mode='reflect') / 100.0  # Significantly reduce laplace effect
    noise = np.random.normal(0, 0.001, image.shape) * 255  # Significantly reduce noise
    noisy_image = image + noise
    transformed_image = 0.98 * image + 0.01 * laplacian_image + 0.01 * noisy_image  # Minimize transformation effects
    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
    return transformed_image

# Function to resize image while maintaining quality using Bicubic Interpolation
def resize_image_with_quality(image, target_size):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_image

# Function to load and process external images while retaining original size
def load_and_preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    img = load_img(image_path)
    img_array = img_to_array(img)
    original_shape = img_array.shape[:2]  # Save original dimensions without channels
    resized_image = resize_image_with_quality(img_array, (224, 224))  # Resize to larger size to retain details
    resized_image = resized_image.astype('float32') / 255.0  # Normalize the image
    return resized_image, original_shape

# Paths to external images
image_paths = [
    r'C:\Users\Lenovo\Desktop\jaguar.jpeg',
    r'C:\Users\Lenovo\Desktop\images.jpeg',
    r'C:\Users\Lenovo\Desktop\tree.jpeg'
]

# Load and process external images while retaining original dimensions
external_images_info = [load_and_preprocess_image(image_path) for image_path in image_paths]
external_images_info = [info for info in external_images_info if info is not None]

# Check if any images were loaded
if not external_images_info:
    print("No images were loaded. Please check your image paths.")
else:
    external_images, original_shapes = zip(*external_images_info)
    external_images = np.array(external_images)

    # Apply texture transformations
    external_images_transformed = np.array([apply_texture_transformations(img * 255) / 255.0 for img in external_images])

    # Resize transformed images to their original size
    external_images_transformed_resized = []
    for i, transformed_image in enumerate(external_images_transformed):
        original_shape = original_shapes[i]  # Extract original dimensions
        transformed_resized = resize_image_with_quality(transformed_image * 255, original_shape[::-1])  # Note CV2 dimensions (width x height)
        external_images_transformed_resized.append(transformed_resized)

    # Define the prediction function
    @tf.function
    def model_predict(model, input_data):
        return model(input_data, training=False)

    # Conduct predictions
    predictions = model_predict(model, external_images_transformed)

    # Display results
    for i, image_path in enumerate(image_paths):
        if os.path.exists(image_path):
            plt.figure(figsize=(10, 5))

            # Display the original image
            plt.subplot(1, 2, 1)
            original_img = load_img(image_path)
            plt.imshow(original_img)
            plt.title('Original Image')
            plt.axis('off')

            # Display the poisoned image
            plt.subplot(1, 2, 2)
            poisoned_img = external_images_transformed_resized[i]
            plt.imshow(poisoned_img.astype(np.uint8))
            plt.title('Poisoned Image')
            plt.axis('off')

            plt.suptitle(f'Prediction: {np.argmax(predictions[i])}')
            plt.show()


# In[63]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from scipy.ndimage import gaussian_filter, laplace

# تحميل النموذج المدرب
model = load_model('texture_transformed_model.h5')

# دالة لتحويلات النسيج مع تقليل التأثيرات
def apply_texture_transformations(image):
    blurred_image = gaussian_filter(image, sigma=0.05)
    laplacian_image = laplace(blurred_image, mode='reflect') / 100.0
    noise = np.random.normal(0, 0.001, image.shape) * 255
    noisy_image = image + noise
    transformed_image = 0.98 * image + 0.01 * laplacian_image + 0.01 * noisy_image
    transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
    return transformed_image

# دالة لإعادة تشكيل الصورة مع الحفاظ على الجودة باستخدام Bicubic Interpolation
def resize_image_with_quality(image, target_size):
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_image

# دالة لتحميل ومعالجة الصور الخارجية مع الاحتفاظ بحجمها الأصلي
def load_and_preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    img = load_img(image_path)
    img_array = img_to_array(img)
    original_shape = img_array.shape[:2]
    resized_image = resize_image_with_quality(img_array, (224, 224))
    resized_image = resized_image.astype('float32') / 255.0
    return resized_image, original_shape

# مسارات الصور الخارجية
image_paths = [
    r'C:\Users\Lenovo\Desktop\jaguar.jpeg',
    r'C:\Users\Lenovo\Desktop\images.jpeg',
    r'C:\Users\Lenovo\Desktop\tree.jpeg'
]

# تحميل ومعالجة الصور الخارجية مع الحفاظ على الأبعاد الأصلية
external_images_info = [load_and_preprocess_image(image_path) for image_path in image_paths]
external_images_info = [info for info in external_images_info if info is not None]

# التحقق مما إذا كانت هناك صور تم تحميلها
if not external_images_info:
    print("No images were loaded. Please check your image paths.")
else:
    external_images, original_shapes = zip(*external_images_info)
    external_images = np.array(external_images)

    # تطبيق التحويلات الملمسية
    external_images_transformed = np.array([apply_texture_transformations(img * 255) / 255.0 for img in external_images])

    # إعادة تشكيل الصور المسممة إلى حجمها الأصلي
    external_images_transformed_resized = []
    for i, transformed_image in enumerate(external_images_transformed):
        original_shape = original_shapes[i]
        transformed_resized = resize_image_with_quality(transformed_image * 255, original_shape[::-1])
        external_images_transformed_resized.append(transformed_resized)

    # إجراء التنبؤ
    predictions = model.predict(external_images_transformed)

    # عرض النتائج
    for i, image_path in enumerate(image_paths):
        if os.path.exists(image_path):
            plt.figure(figsize=(10, 5))

            # عرض الصورة الأصلية
            plt.subplot(1, 2, 1)
            original_img = load_img(image_path)
            plt.imshow(original_img)
            plt.title('Original Image')
            plt.axis('off')

            # عرض الصورة المسممة
            plt.subplot(1, 2, 2)
            poisoned_img = external_images_transformed_resized[i]
            plt.imshow(poisoned_img.astype(np.uint8))
            plt.title('Poisoned Image')
            plt.axis('off')

            plt.suptitle(f'Prediction: {np.argmax(predictions[i])}')
            plt.show()


# In[ ]:


from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import cv2
from scipy.ndimage import gaussian_filter, laplace

app = Flask(__name__)

# Load your model
model = load_model('texture_transformed_model.h5')

# Define functions for image processing
def apply_texture_transformations(image):
    # Your texture transformation function here
    pass

def resize_image_with_quality(image, target_size):
    # Your image resizing function here
    pass

def load_and_preprocess_image(image_path):
    # Your image loading and preprocessing function here
    pass

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']
    image_path = f'/tmp/{file.filename}'
    file.save(image_path)

    # Process the uploaded image
    image, _ = load_and_preprocess_image(image_path)
    transformed_image = apply_texture_transformations(image)

    # Make predictions
    prediction = model.predict(np.expand_dims(transformed_image, axis=0))

    # Decode prediction (assuming your model outputs categorical predictions)
    predicted_class = np.argmax(prediction)

    # Return the result
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:


python app.py
#http://127.0.0.1:5000/predict


# In[ ]:




