import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical

import matplotlib.pyplot as plt
data=fashion_mnist
(train_ind,train_dep),(test_ind,test_dep)=data.load_data()
print(train_ind.shape[1:])
print(train_dep.shape)
print(test_dep.shape)
print(train_ind[0])#pixe;s of 1 row of ind

#try and error to get no.s and its values in class_names
print(train_dep[1])#numslike 1 2 3 4
len(train_ind[1])
plt.imshow(train_ind[1])#img of particular pixel

class_names=['T-shirt/top' , 'Trouser' , 'Pullover' , 'Dress' , 'Coat' , 'Scandals' , 'dress' , 'Sneakers' , 'Bag' , 'Ankle-boot']

train_ind=train_ind.astype('float32')
test_ind=test_ind.astype('float32')

train_ind /= 255 #scale
test_ind /= 255 #scale

train_ind=train_ind.reshape(train_ind.shape[0],28,28,1)
train_dep=to_categorical(train_dep)

mod=Sequential()
mod.add(Conv2D(32 ,(3,3), padding='same',input_shape=(28,28,1),activation='relu'))
mod.add(MaxPool2D(pool_size = (2,2)))
mod.add(Conv2D(32 ,(3,3),activation='relu'))
mod.add(MaxPool2D(pool_size = (2,2)))
mod.add(Conv2D(32 ,(3,3),activation='relu'))
mod.add(MaxPool2D(pool_size = (2,2)))
mod.add(Flatten())
mod.add(Dense(units = 128,activation = 'relu'))
mod.add(Dense(units = 10,activation = 'softmax'))

mod.compile(optimizer='adam' ,loss='categorical_crossentropy' , metrics=['accuracy'])
mod.fit(train_ind ,train_dep ,epochs=2)

mod.save_weights(r'C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\dogs_cats_model.h5')

mod.load_weights(r'C:\Users\omkar desai\OneDrive\Desktop\Artificial Inteligence\dogs_cats_model.h5')

test_ind = test_ind.reshape(test_ind.shape[0],28,28,1)
test_dep = to_categorical(test_dep)
test_loss,test_accuracy=mod.evaluate(test_ind,test_dep)

print('Test accuracy:',test_accuracy)

train_ind[0].shape

predictions = mod.predict(test_ind[0])
test_ind[0].shape
predictions[0]
















data=fashion_mnist
(train_ind,train_den),(test_inde,test_dent)=data.load_data()
print(train_ind.shape[1:])
print(train_den.shape)
print(test_dent.shape)
print(train_ind[0])#pixe;s of 1 row of ind

#try and error to get no.s and its values in class_names
print(train_den[1])#numslike 1 2 3 4
len(train_ind[1])
plt.imshow(train_ind[1])
print(train_ind.shape)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_ind = train_ind.astype('float32')
test_inde = test_inde.astype('float32')

train_ind /= 255 #scale
test_inde /= 255
train_ind= train_ind.reshape(train_ind.shape[0], 28,28, 1)
train_den = to_categorical(train_den)

mod=Sequential()
mod.add(Conv2D(32, (3,3), padding='same',input_shape=(28,28,1)))
mod.add(MaxPool2D(pool_size = (2, 2)))
mod.add(Conv2D(32,(3,3),activation='relu'))
mod.add(MaxPool2D(pool_size = (2, 2)))
mod.add(Conv2D(32,(3,3),activation='relu'))
mod.add(MaxPool2D(pool_size = (2, 2)))
mod.add(Conv2D(32,(3,3),activation='relu'))
mod.add(MaxPool2D(pool_size = (2, 2)))
mod.add(Flatten())
mod.add(Dense(units = 128, activation = 'relu'))
mod.add(Dense(units=10,activation="softmax"))


mod.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'] )
mod.fit(train_ind,train_den,epochs=2)

test_inde= test_inde.reshape(test_inde.shape[0], 28,28, 1)
test_dent=to_categorical(test_dent)
test_loss, test_acc = mod.evaluate(test_inde, test_dent)

print('Test accuracy:', test_acc)

predictions = mod.predict(test_inde[0])
#test_inde[0].shape
predictions[0]

np.argmax(predictions[0])

test_labels[0]




