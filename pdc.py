import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
import numpy as np
import os
#Set All Constants
BATCH_SIZE=32
IMAGE_SIZE=256
CHANNELS=3
EPOCHS=50

#Import Data into tensorflow dataset object
#image_dataset_from_directory(main_directory, labels='inferred') will
# return a tf.data.Dataset that yields batches of images from the subdirectories
# class_a and class_b, together with labels 0 and 1 (0 corresponding to class_a and 1 corresponding to class_b).
dataset=tf.keras.preprocessing.image_dataset_from_directory(
        "PlantVillage" , #Name of the Data Set
        #Seed=123,
        shuffle=True, # Random Data display
        image_size=(IMAGE_SIZE,IMAGE_SIZE),
        batch_size=BATCH_SIZE
)

class_names=dataset.class_names
print(class_names)

for image_batch,labels_batch in dataset.take(1):
        print(image_batch.shape)
        print(labels_batch.numpy())
#Visualize Some of the Images From our DataSet
plt.figure(figsize=(10,10))

for image_batch,labels_batch in dataset.take(1):
        for i in range(12):
                ax=plt.subplot(3,4,i+1)
                plt.imshow(image_batch[i].numpy().astype("uint8"))
                plt.title(class_names[labels_batch[i]])
                plt.axis("off")

#Function to Split Dataset
#Dataset Should be bifurcated into 3 subsets, namely:
#1) Training:Dataset to be used while Training
#2) Validation:Dataset to be tested against while Training
#3) Test:Dataset to be against after we trained a model

a=len(dataset)
print(a)


train_size=0.8
b=len(dataset)*train_size
print(b)

c=test_ds=dataset.take(54)
len(test_ds)
print(c)

p=train_ds=dataset.skip(54)
len(train_ds)
print(p)

val_size=0.1
q=len(dataset)*val_size
print(q)

val_ds=test_ds.take(6)
r=len(val_ds)
print(r)

test_ds=test_ds.skip(6)
k=len(test_ds)
print(k)

def get_dataset_partitions_tf(ds,train_split=0.8,val_split=0.1,test_split=0.1,shuffle=True,shuffle_size=1000):
        assert (train_split+test_split+val_split)==1
        ds_size=len(ds)

        if shuffle:
                ds=ds.shuffle(shuffle_size,seed=12)

        train_size=int(train_split*ds_size)
        val_size=int(val_split*ds_size)

        train_ds=ds.take(train_size)
        val_ds=ds.skip(train_size).take(val_size)
        test_ds=ds.skip(train_size).skip(val_size)

        return train_ds,val_ds,test_ds

train_ds,val_ds,test_ds=get_dataset_partitions_tf(dataset)
print(len(train_ds))
print(len(val_ds))
print(len(test_ds))

#Cache,Shuffle and Prefetch the dataset
test_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

#Building the Model
#Creating a Layer for Resizing and Normalization
#Before we feed our images to Network,we Should be resizing it to the desired size.Moreover to improve model Performance
#we should normalize the image pixel value (Keeping them in range 0 & 1 by dividing by 256).This should happen while
#training as well as inference.Hence we can add that as a layer in our sequential model.

resize_and_rescale=tf.keras.Sequential([
        layers.experimental.preprocessing.Resizing(IMAGE_SIZE,IMAGE_SIZE),
        layers.experimental.preprocessing.Rescaling(1./255),
])

#Data Augmentation
#Data Augmentation is Needed when we have less data,this boosts the accuracy of our model by augmenting the data

data_augmentation=tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),

])


#Applying Data Augmentation to Train DataSet

train_ds=train_ds.map(
        lambda x,y:(data_augmentation(x,training=True),y)
).prefetch(buffer_size=tf.data.AUTOTUNE)

#Model Architecture
#we use CNN coupled with a SoftMax Activation in the output layer.we also add the initial layers for resizing,normalization and Data_Augmentation

input_shape=(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,CHANNELS)
n_classes=3
model=models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,kernel_size=(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dense(n_classes,activation='softmax')
])

model.build(input_shape=input_shape)
print(model.summary())

#Compiling the Model
#we use adam Optimizer, SparseCategoricalCrossentropy for losses,accuracy as a metric

model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']

)

history=model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        validation_data=val_ds,
        verbose=1,
        epochs=50,
)
print(history)

#Scores is just a list containing loss and accuracy valu
scores=model.evaluate(test_ds)
print(scores)

#Plotting the Accuracy and Loss Curves

print(history.params)
print(history.history.keys()) #loss,accuracy,val loss etc are a python list containing values of loss,accuracy etc at the end of each epoch
print(type(history.history['loss']))

print(history.history['loss'][:5]) # show loss for first 5 epochs

acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(EPOCHS),acc,label='Training Accuracy')
plt.plot(range(EPOCHS),val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1,2,2)
plt.plot(range(EPOCHS),loss,label='Training Loss')
plt.plot(range(EPOCHS),val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Run Prediction on a Sample Image

for images_batch,labels_batch in test_ds.take(1):
        first_image=images_batch[0].numpy().astype('uint8')
        first_label=labels_batch[0].numpy()

        print("First image to predict")
        plt.imshow(first_image)
        print("actual label:",class_names[first_label])

        batch_prediction=model.predict(images_batch)
        print("predicted label:",class_names[np.argmax(batch_prediction[0])])

#Write a function for inference

def predict(model,img):
        img_array=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
        img_array=tf.expand_dims(img_array,0)

        predictions=model.predict(img_array)

        predicted_class=class_names[np.argmax(predictions[0])]
        confidence=round(100*(np.max(predictions[0])),2)
        return predicted_class,confidence

#Now run inference on few sample images

plt.figure(figsize=(15,15))
for images,labels in test_ds.take(1):
        for i in range(9):
                ax=plt.subplot(3,3,i+1)
                plt.imshow(images[i].numpy().astype("uint8"))
                predicted_class,confidence=predict(model,images[i].numpy())
                actual_class=class_names[labels[i]]
                plt.title(f"Actual:{actual_class},\n Predicted: {predicted_class},\n Confidence: {confidence}%")
                plt.axis("off")

#Saving the Model
model_version=1
model.save(f"Plant disease classifier.keras")

#model_version=max([int (i) for i in os.listdir("../Models")+[0]])+1
#model.save(f"../Models/{model_version}")