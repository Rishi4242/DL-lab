#To implement MLP on MNIST dataset using keras 
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt


#Load data
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
# print(f"x_train.shape:{x_train.shape}\n")
# plt.imshow(x_train[0])
# plt.show()ubjb


#Preprocessing 
# print(y_train[0])
# print(type(y_train[0]))
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)   #to convert any datatype to categorical
# print(y_train[0])
# print(type(y_train[0]))


#Build the Architecture
model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(units=10,activation='softmax'))


#Compile
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])


#Build the model 
res=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_test,y_test))
# print(res.history.keys())
# print(res.history.items())

#Evaluate
loss,accuracy=model.evaluate(x_test,y_test)
print(f"\nTest loss:{loss},\nTest Accuracy:{accuracy}\n")


# #Visualization
plt.plot(res.history['loss'],label="Train Loss",color="blue")
plt.plot(res.history['val_loss'],label="Validation Loss",color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("E vs L")
plt.show()


# #Visualization
plt.plot(res.history['accuracy'],label="Train Accuracy",color="blue")
plt.plot(res.history['val_accuracy'],label="Validation Accuracy",color="red")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("E vs A")
plt.show()
