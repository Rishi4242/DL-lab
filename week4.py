# Load the libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocessing
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the architecture
model = Sequential()
model.add(Flatten(input_shape=(32, 32, 3)))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
result = model.fit(x_train, y_train,
                   epochs=50,
                   batch_size=64,
                   validation_split=0.2)

# Evaluate model
loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)

# Predictions
pred = model.predict(x_test)
predicted_labels = np.argmax(pred, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Show sample predictions
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
plt.show()

# Plot loss
plt.plot(result.history['loss'], label='train loss', color='blue')
plt.plot(result.history['val_loss'], label='validation loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.plot(result.history['accuracy'], label='train accuracy', color='blue')
plt.plot(result.history['val_accuracy'], label='validation accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Epoch vs Accuracy')
plt.legend()
plt.show()
