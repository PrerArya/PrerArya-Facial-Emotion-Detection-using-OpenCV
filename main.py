from keras.preprocessing.image import  ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import os

#path for dataset
train_path = 'dataset/train'
test_path = 'dataset/test'

# ImageDataGenerator for train and test(Image Augmentation)
train_datagen = ImageDataGenerator(rescale=1./255,#normalization
                                   rotation_range=30,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

#flow_from_directory for train and test
train_generator = train_datagen.flow_from_directory(train_path,
                                                    color_mode='grayscale',
                                                    target_size=(48,48),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

test_generator = test_datagen.flow_from_directory(test_path,
                                                  color_mode='grayscale',
                                                  target_size=(48,48),
                                                  batch_size=32,
                                                  class_mode='categorical',
   
                                                  shuffle=True)
#creating labels for images
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
img,label = train_generator.next()

# Load CNN Model
model = Sequential()

#INPUT LAYER
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))

#HIDDEN LAYER
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))

#OUTPUT LAYER
model.add(Dense(7,activation='softmax'))
model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

# Train the model
num_train_images = 0 
for root,dirs,files in os.walk(train_path):
    #counting number of images in train folder
    num_train_images += len(files)

num_test_images = 0
for root,dirs,files in os.walk(test_path):
    #counting number of images in test folder
    num_test_images += len(files)

final_model = model.fit_generator(train_generator,
                                  steps_per_epoch=num_train_images//32,
                                  epochs=30,
                                  validation_data=test_generator,
                                  validation_steps=num_test_images//32)

# Save the model
model.save('emotion_detection.h5')
