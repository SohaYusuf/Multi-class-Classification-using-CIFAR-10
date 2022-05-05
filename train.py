"""

# ECSE 6850 Programming Assignment 3 | Soha Yusuf (RIN: 662011092)

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf



# initialize the hyper-parameters
BATCH_SIZE = 512        # batch size
EPOCHS = 400            # number of epochs
alpha = 0.0001          # learning rate



# load training data
X_train = np.load('training_data.npy')
Y_train = np.load('training_label.npy')
# load testing data
X_test = np.load('testing_data.npy')
Y_test = np.load('testing_label.npy')

# normalize the images
X_train = np.reshape(X_train, (50000,32,32,3))/255.0
X_test = np.reshape(X_test, (5000,32,32,3))/255.0

# convert labels to one_hot vectors
Y_train = tf.reshape((tf.one_hot(Y_train.astype(np.int32), depth=10)) , (50000,10))       
Y_test = tf.reshape((tf.one_hot(Y_test.astype(np.int32), depth=10)) , (5000,10))

print('X_train shape: ',X_train.shape, 'Y_train shape: ',Y_train.shape)
print('X_test shape: ',X_test.shape, 'Y_test shape: ',Y_test.shape)



def my_model(input_shape):
    # initialize regularization
    reg = tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)
    # initialize the weights randomly by Xavier Normal initialization
    initializer = tf.keras.initializers.GlorotNormal()
    # input shape for one image is (32,32,3)
    input_img = tf.keras.Input(shape=input_shape)
    # Conv1: 16 filters 5x5 with stride = 1 and ReLU activation
    Conv1 = tf.keras.layers.Conv2D(16, (5, 5) , (1, 1), padding='valid', activation='relu', kernel_initializer=initializer)(input_img)
    # Batch normalization
    B1 = tf.keras.layers.BatchNormalization()(Conv1)
    # Maxpool1: 2x2 with stride 2
    Maxpool1 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(B1)
    # Dropout with dropout percentage of 25%
    D1 = tf.keras.layers.Dropout(0.25)(Maxpool1)
    # Conv2: 32 filters 5x5 with stride = 1 and ReLU activation
    Conv2 = tf.keras.layers.Conv2D(32, (5, 5) , (1, 1), padding='valid', activation='relu', kernel_initializer=initializer)(D1)
    # Batch normalization
    B2 = tf.keras.layers.BatchNormalization()(Conv2)
    # Maxpool2: 2x2 with stride 2
    Maxpool2 = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(B2)
    # Dropout with dropout percentage of 25%
    D2 = tf.keras.layers.Dropout(0.25)(Maxpool2)
    # Conv3: 64 filters 3x3 with stride = 1 and ReLU activation
    Conv3 = tf.keras.layers.Conv2D(64, (3, 3) , (1, 1), padding='valid', activation='relu', kernel_initializer=initializer)(D2)
    # Batch normalization
    B3 = tf.keras.layers.BatchNormalization()(Conv3)
    # Flatten (576,1)
    F1 = tf.keras.layers.Flatten()(B3)
    # Dropout with dropout percentage of 25%
    D3 = tf.keras.layers.Dropout(0.25)(F1)
    # Fully connected layer with 500 nodes
    F2 = tf.keras.layers.Dense(units=500, activation='relu', kernel_regularizer=reg)(D3)
    # Dropout with dropout percentage of 25%
    D4 = tf.keras.layers.Dropout(0.25)(F2)
    # Fully connected layer with 10 nodes and softmax activation function
    outputs = tf.keras.layers.Dense(units=10, activation='softmax', kernel_regularizer=reg)(D4)
    # define the model with input and output
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model



# For one training example, print the model architecture, feature map shapes and number of parameters
model = my_model(X_train[0].shape)
print(model.summary())



# define the optimizer, loss and metric
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=alpha),
    loss='categorical_crossentropy',
    metrics=['acc']
)



# train the model
history = model.fit(
    X_train,
    Y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test,Y_test),
    shuffle=True
)



# value of training loss at the end of training
final_train_loss = history.history['loss'][-1]
# value of testing loss at the end of training
final_test_loss = history.history['val_loss'][-1]
# value of training accuracy at the end of training
final_train_acc = history.history['acc'][-1]
# value of testing loss at the end of training
final_test_acc = history.history['val_acc'][-1]



df_loss_acc = pd.DataFrame(history.history)
# plot the training and testing loss
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'Training Loss','val_loss':'Testing Loss'},inplace=True)
# plot the training and testing accuracy
df_acc= df_loss_acc[['acc','val_acc']]
df_acc.rename(columns={'acc':'Training accuracy (final={:1.3f})'.format(final_train_acc),'val_acc':'Test accuracy (final={:1.3f})'.format(final_test_acc)},inplace=True)
# Title of loss curves
df_loss.plot(title='Model loss',figsize=(8,8)).set(xlabel='Epoch',ylabel='Loss')
plt.grid(True)
# Title of accuracy curves
df_acc.plot(title='Model Accuracy',figsize=(8,8)).set(xlabel='Epoch',ylabel='Accuracy')
plt.grid(True)



# evaluate the model
def error(Y_true,Y_pred):
    # convert y_hat to one-hot vectors
    Y_pred = tf.one_hot(tf.math.argmax(Y_pred, 1), 10, dtype=tf.float32)
    # classification error for each digit
    errors = np.zeros(10)
    for each_class in range(Y_true.shape[1]):
        num_correct = tf.reduce_sum(tf.math.multiply(Y_pred[:, each_class], Y_true[:, each_class]))
        num_labels = tf.reduce_sum(Y_true[:, each_class])
        errors[each_class] = 1 - (num_correct / num_labels)
    avg_class_error = np.average(errors)
    return avg_class_error, errors



# predict on the test set
Y_pred = model.predict(X_test)
for i in range(10):
    avg_class_error, errors = error(Y_test,Y_pred)
    print(f'Classification Error for Class {i}: ', errors[i])

print('Average Classification Error: ', avg_class_error)

print('Final Training Loss = ',final_train_loss)
print('Final Training Accuracy = ',final_train_acc)
print('Final Test Loss = ',final_test_loss)
print('Final Test Accuracy = ',final_test_acc)



# visualize the filter in first convolutional layer
layer = model.layers[1]
filters, biases = layer.get_weights()
print(layer.name , filters.shape)
# normalize the filters for visualization
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
a = 4
b = 8
c = 1
fig = plt.figure(figsize=(20,20))
for i in range(16):
    plt.subplot(a, b, c)
    plt.title('Filter {}'.format(i+1))
    plt.imshow((filters[:,:,:,i] * 255).astype(np.uint8))
    c = c + 1
plt.show()



# save the model
model.save("trained_model.h5")



# load the model and evaluate of test set
new_model = tf.keras.models.load_model('trained_model.h5')
loss, acc = new_model.evaluate(X_test, Y_test, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
