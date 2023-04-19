
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data import Data

PLAYLIST = 'ranked-duels'
DIRECTORY = 'results/neural_network/'
FILECOUNT = 40

class NeuralNetwork():

    '''
    A class for using various neural networks on the input data
    '''

    def __init__(self, X_train, y_train, X_test, y_test):
        '''
        Turn the input data into numpy arrays
        '''
        self.augmentFlag = 0
        self.X_train_x = np.array(X_train[0])
        self.X_train_y = np.array(X_train[1])
        self.X_train_z = np.array(X_train[2])
        self.y_train_x = np.array(y_train[0])
        self.y_train_y = np.array(y_train[1])
        self.y_train_z = np.array(y_train[2])
        self.X_test_x = np.array(X_test[0])
        self.X_test_y = np.array(X_test[1])
        self.X_test_z = np.array(X_test[2])
        self.y_test_x = np.array(y_test[0])
        self.y_test_y = np.array(y_test[1])
        self.y_test_z = np.array(y_test[2])


    def train(self):
        '''
        This method runs the base functions for training
        '''
        #Bias
        self.file = f'{DIRECTORY}neural_network.txt'
        self.trainModel(self.X_train_x, self.y_train_x, self.X_test_x, self.y_test_x, 'w', 'X')
        self.trainModel(self.X_train_y, self.y_train_y, self.X_test_y, self.y_test_y, 'a', 'Y')
        self.trainModel(self.X_train_z, self.y_train_z, self.X_test_z, self.y_test_z, 'a', 'Z')

        #Augment the features to absorb the bias term
        self.file = f'{DIRECTORY}neural_network_augmented.txt'
        self.augmentFlag = 1
        self.trainModel(self.X_train_x, self.y_train_x, self.X_test_x, self.y_test_x, 'w', 'X')
        self.trainModel(self.X_train_y, self.y_train_y, self.X_test_y, self.y_test_y, 'a', 'Y')
        self.trainModel(self.X_train_z, self.y_train_z, self.X_test_z, self.y_test_z, 'a', 'Z')


    def trainModel(self, X_train, y_train, X_test, y_test, format, axis):
        '''
        This method will do the actual training
        '''
        #X_train, y_train = self.normalizeInputs(X_train, y_train)
        
        # Define the neural network architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(81, input_shape=[2], activation='relu'),
            tf.keras.layers.Dense(27, activation='relu'),
            tf.keras.layers.Dense(9, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        if self.augmentFlag:
            X_train, X_test = self.augment(X_train, X_test)

            # Define the neural network architecture for the augmented data
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(81, input_shape=[3], activation='relu'),
                tf.keras.layers.Dense(27, activation='relu'),
                tf.keras.layers.Dense(9, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

        # Compile the model with an optimizer and a loss function
        model.compile(optimizer='adam', loss='mse')

        # Train the model on the training data
        history = model.fit(X_train, y_train, epochs=self.epochs, steps_per_epoch=self.steps, validation_data=(X_test, y_test), verbose=1)

        # Evaluate the model stats
        trainingLoss = model.evaluate(X_train, y_train, verbose=0)

        testLoss = model.evaluate(X_test, y_test, verbose=0)

        file = open(self.file, format)
        file.write(f"\n\nClosed Form {axis}")
        file.write(f"\n\tTraining Loss: {trainingLoss}\n\tTest Loss: {testLoss}")
        file.close()

        augmented = ''
        if self.augmentFlag:
            augmented = '_augmented'

        # plot loss
        plt.figure()
        plt.plot(history.history['loss'], label='train_loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'{DIRECTORY}training_loss{augmented}_{axis}.jpg')

        plt.figure()
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.title('Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(f'{DIRECTORY}test_loss{augmented}_{axis}.jpg')


    def augment(self, X_train, X_test):
        '''
        Augment the given input features
        '''
        X_train = np.concatenate((np.ones([X_train.shape[0], 1]), X_train), axis=1)
        X_test = np.concatenate((np.ones([X_test.shape[0], 1]),  X_test), axis=1)
        
        return X_train, X_test


    def normalizeInputs(self, X_train, y_train):
        '''
        This method will normalize the inputs to scale down
        '''

        X_train_norm = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

        y_train_norm = (y_train - np.mean(y_train)) / np.std(y_train)

        return X_train_norm, y_train_norm
    

    def setParams(self, epochs, steps):
        '''
        Set the parameters for the neural network for different experiments
        '''
        self.epochs = epochs
        self.steps = steps


def main():
    d = Data(PLAYLIST, FILECOUNT, 1000)
    X_train, y_train, X_test, y_test = d.getData()

    nn = NeuralNetwork(X_train, y_train, X_test, y_test)
    nn.setParams(50, 250)
    nn.train()


if __name__ == "__main__":
    main()