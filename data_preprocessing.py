# data_preprocessing.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_base_path = 'Datasets/'

gender_train_directory = os.path.join(dataset_base_path, 'Gender dataset/Train')
gender_validation_directory = os.path.join(dataset_base_path, 'Gender dataset/Validation')
gender_test_directory = os.path.join(dataset_base_path, 'Gender dataset/Test')

age_train_directory = os.path.join(dataset_base_path, 'Age dataset/Train')
age_validation_directory = os.path.join(dataset_base_path, 'Age dataset/Validation')
age_test_directory = os.path.join(dataset_base_path, 'Age dataset/Test')

# Creating ImageDataGenerator for training, validation and testing
train_imagedatagenerator = ImageDataGenerator(
                                            rescale = 1./255,
                                            rotation_range = 30,
                                            width_shift_range = 0.2,
                                            height_shift_range = 0.2,
                                            shear_range = 0.2,
                                            zoom_range = 0.2,
                                            horizontal_flip = True,
                                            fill_mode = 'nearest')
# bez augmentacji zeby byly consistent wyniki tego jak sie uczy model, tylko resize, zeby procesowalo w ten sam sposob
validation_imagedatagenerator = ImageDataGenerator(rescale = 1./255)
test_imagedatagenerator = ImageDataGenerator(rescale = 1./255)

# Loading data

# class mode - binary dla gender bo albo male albo female

print("Gender train dataset")
gender_train_idg = train_imagedatagenerator.flow_from_directory(
                                            gender_train_directory,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='binary',
                                            shuffle=False) # na wszelki wypadek skoro sie takie dziwne rzeczy dzieja
print("Gender train data loaded.")
print("Gender validation dataset")
gender_validation_idg = validation_imagedatagenerator.flow_from_directory(
                                            gender_validation_directory,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='binary')
print("Gender validation data loaded.")
print("Gender test dataset")
gender_test_idg = test_imagedatagenerator.flow_from_directory(
                                            gender_test_directory,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='binary')
print("Gender test data loaded.")

# class mode - categorical dla age bo sa kategorie wiekowe /// albo sparse???

print("Age train dataset")
age_train_idg = train_imagedatagenerator.flow_from_directory(
                                            age_train_directory,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')
print("Age train data loaded.")
print("Age validation dataset")
age_validation_idg = validation_imagedatagenerator.flow_from_directory(
                                            age_validation_directory,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')
print("Age validation data loaded.")
print("Age test dataset")
age_test_idg = test_imagedatagenerator.flow_from_directory(
                                            age_test_directory,
                                            target_size=(224,224),
                                            batch_size=32,
                                            class_mode='categorical')
print("Age test data loaded.")
##### flow_from_directory() automatycznie printuje logi podczas skanowania directories #####

print("Gender class distribution in training data:")
print("Gender class indices: ", gender_train_idg.class_indices)
print("Gender labels: ", gender_train_idg.classes[:10]) # First 10 labels # czemu printuje same kobiety???
print("Number of images - training data - male:", len(os.listdir(os.path.join(gender_train_directory, 'male'))))
print("Number of images - training data - female:", len(os.listdir(os.path.join(gender_train_directory, 'female'))))

print("Age class distribution in training data:")
print("Age class indices: ", age_train_idg.class_indices)
print("Age image count per category: ", np.unique(age_train_idg.classes, return_counts=True))

########JEZU CZEMU TO WYZEJ NIE DZIALAAAAAAAAAAAAAAAA
batch_images, batch_labels = next(gender_test_idg)
print("Sample labels printed directly from directory:", batch_labels[:10]) # tylko na check, bo to wyzej cos dziwnie printuje

figure, axes=plt.subplots(1,10, figsize=(15,5))
for i in range(10):
    axes[i].imshow(batch_images[i])
    axes[i].title.set_text(f"Label: {batch_labels[i]}")
    axes[i].axis("off")
plt.show()

