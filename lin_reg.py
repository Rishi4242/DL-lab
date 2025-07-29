import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

#data
X=np.linspace(0,5,1000)
# X.reshape(1,-1)
# print(X.shape)
# print(X)
y=5*X+7+np.random.randn(1000)
# print(y)

#build the architecture
model=Sequential()
model.add(Dense(units=1,input_shape=(1,),activation='linear'))
# model.add(Dense(units=1,activation='linear'))  from second layer we should not use the input_shape

#compile
model.compile(optimizer='sgd',loss='mse')

#build the model
result=model.fit(X,y,epochs=50)#first take 1 as epochs and next take 10 like wise increase

#Make predictions
predict=model.predict(X)

#Visualization
# plt.scatter(X,y,label="original data",color='blue')
plt.plot(result.history['loss'])
# plt.plot(X,predict,label='Predcitions',color='red')
plt.xlabel("X")
plt.ylabel("y")
# plt.legend()
plt.title("Loss Over Epochs")
plt.show()

