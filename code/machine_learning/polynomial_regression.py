
import numpy as np
import data
import matplotlib.pyplot as plt

PLAYLIST = 'ranked-duels'
FILECOUNT = 40

class PolynomialRegression:

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


    def closedForm(self):
        '''
        Perform closed form polynomial regression
        '''

        #Bias
        self.file = 'polynomial_regression.txt'
        self.trainClosedForm(self.X_train_x, self.X_test_x, self.y_train_x, self.y_test_x, 'w', 'X')
        self.trainClosedForm(self.X_train_y, self.X_test_y, self.y_train_y, self.y_test_y, 'a', 'Y')
        self.trainClosedForm(self.X_train_z, self.X_test_z, self.y_train_z, self.y_test_z, 'a', 'Z')

        #Augment the features to absorb the bias term
        self.file = 'polynomial_regression_augmented.txt'
        X_train, X_test = self.augment(self.X_train_x, self.X_test_x)
        self.trainClosedForm(X_train, X_test, self.y_train_x, self.y_test_x, 'w', 'X')

        X_train, X_test = self.augment(self.X_train_y, self.X_test_y)
        self.trainClosedForm(X_train, X_test, self.y_train_y, self.y_test_y, 'a', 'Y')

        X_train, X_test = self.augment(self.X_train_z, self.X_test_z)
        self.trainClosedForm(X_train, X_test, self.y_train_z, self.y_test_z, 'a', 'Z')


    def gradientDescent(self):
        '''
        This method will peform gradient descent on the data
        '''
        #Bias
        self.file = 'gradient_descent.txt'
        self.trainGradientDescent(self.X_train_x, self.X_test_x, self.y_train_x, self.y_test_x, 'w', 'X')
        print("Finished gradient descent on x, unaugmented")
        self.trainGradientDescent(self.X_train_y, self.X_test_y, self.y_train_y, self.y_test_y, 'a', 'Y')
        self.trainGradientDescent(self.X_train_z, self.X_test_z, self.y_train_z, self.y_test_z, 'a', 'Z')

        #Augment the features to absorb bias term
        self.file = 'gradient_descent_augmented.txt'
        self.augmentFlag = 1
        X_train, X_test = self.augment(self.X_train_x, self.X_test_x)
        self.trainGradientDescent(X_train, X_test, self.y_train_x, self.y_test_x, 'w', 'X')

        X_train, X_test = self.augment(self.X_train_y, self.X_test_y)
        self.trainGradientDescent(X_train, X_test, self.y_train_y, self.y_test_y, 'a', 'Y')

        X_train, X_test = self.augment(self.X_train_z, self.X_test_z)
        self.trainGradientDescent(X_train, X_test, self.y_train_z, self.y_test_z, 'a', 'Z')
        
    
    def trainClosedForm(self, X_train, X_test, y_train, y_test, format, axis):
        '''
        Get the weights and statistics for the x axis data and write it to file
        '''
        w = self.getWeights(X_train, y_train)

        #t for training, v for validation
        tMSE, tRMSE, tMAE = self.getStatistics(X_train, y_train, w)
        vMSE, vRMSE, vMAE = self.getStatistics(X_test, y_test, w)

        self.writeStats(tMSE, tRMSE, tMAE, vMSE, vRMSE, vMAE, format, axis)


    def getWeights(self, X, y):
        '''
        Use the closed form method to calculate the w vector
        '''
        XTX = X.T @ X
        XTXinv = np.linalg.inv(XTX)
        XTY = X.T @ y
        w = XTXinv @ XTY

        return w


    def getStatistics(self, X, y, w):
        '''
        Calculate the statistics on the regression data
        '''
        MSE = np.mean((y - np.dot(X, w))**2)
        RMSE = MSE**(0.5)
        MAE = np.mean(np.absolute(y - np.dot(X, w)))

        return MSE, RMSE, MAE
    

    def writeStats(self, tMSE, tRMSE, tMAE, vMSE, vRMSE, vMAE, format, axis):
        '''
        Write the statistics to the specified file
        '''
        file = open(self.file, format)
        file.write(f"\n\nClosed Form {axis}")
        file.write(f"\n\tTraining:\n\t\tMSE: {tMSE}\n\t\tRMSE: {tRMSE}\n\t\tMAE: {tMAE}")
        file.write(f"\n\n\tValidation:\n\t\tMSE: {vMSE}\n\t\tRMSE: {vRMSE}\n\t\tMAE: {vMAE}")
        file.close()


    def augment(self, X_train, X_test):
        '''
        Augment the given input features
        '''
        X_train = np.concatenate((np.ones([X_train.shape[0], 1]), X_train), axis=1)
        X_test = np.concatenate((np.ones([X_test.shape[0], 1]),  X_test), axis=1)
        
        return X_train, X_test
        

    def setGradientParams(self, batchSize, maxEpoch, alpha, decay):
        '''
        This method will set the basic gradient descent parameters
        '''
        self.batchSize = batchSize
        self.maxEpoch = maxEpoch
        self.alpha = alpha
        self.decay = decay


    def predict(self, X, y, w):
        '''
        This method will predict the y_hat given X and w and then get the loss and risk
        '''
        y_hat = np.dot(X, w)

        loss = np.sum((np.square(y - y_hat)))/(2*self.batchSize)

        risk = np.sum(abs(y - y_hat))/(self.batchSize)

        return y_hat, loss, risk
    

    def trainGradientDescent(self, X_train, X_test, y_train, y_test, format, axis):
        '''
        This method will perform gradient descent for the given data
        '''

        X_train, X_test, y_train, y_test = self.normalizeInputs(X_train, X_test, y_train, y_test)

        nTrain = X_train.shape[0]

        w = np.zeros((X_train.shape[1]))

        lossesTrain = []
        risksTest = []

        riskBest = float('inf')
        wBest = None

        for epoch in range(self.maxEpoch):
            print(f"Starting epoch: {epoch}, axis: {axis}")

            lossEpoch = 0

            batches = int(np.ceil(nTrain/self.batchSize))
            for b in range(batches):

                X_batch = X_train[b*self.batchSize: (b+1)*self.batchSize]
                y_batch = y_train[b*self.batchSize: (b+1)*self.batchSize]

                y_hat_batch, lossBatch, _ = self.predict(X_batch, y_batch, w)
                lossEpoch += lossBatch

                dW = np.dot(X_batch.T, y_batch - y_hat_batch) / self.batchSize
                w = w + self.alpha*dW

            tLoss = lossEpoch / self.batchSize
            lossesTrain.append(tLoss)

            y_pred = X_test @ w

            epochRisk = np.mean(abs(y_test - y_pred))
            
            risksTest.append(epochRisk)

            if epochRisk < riskBest:
                wBest = w
                riskBest = epochRisk

        self.writeData(X_test, y_test, wBest, riskBest, lossesTrain, risksTest, axis, format)


    def normalizeInputs(self, X_train, X_test, y_train, y_test):

        X_train_norm = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        X_test_norm = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

        y_train_norm = (y_train - np.mean(y_train)) / np.std(y_train)
        y_test_norm = (y_test - np.mean(y_train)) / np.std(y_train)


        return X_train_norm, X_test_norm, y_train_norm, y_test_norm


    def writeData(self, X_test, y_test, wBest, riskBest, lossesTrain, risksTest, axis, format):
        '''
        This method will write the graphs and data to disk
        '''

        t = np.mean(abs(y_test - np.dot(X_test, wBest)))

        #Save Numbers to file
        file = open(self.file, format)
        file.write(f"\n\nClosed Form {axis}")
        file.write(f"\n\tValidation performance: {riskBest}\n\tTest performance: {t}")
        file.close()

        augmented = ""
        if self.augmentFlag:
            augmented = "_augmented"

        #Plot the training losses
        plt.figure()
        plt.plot(lossesTrain)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.savefig(f'training_loss{augmented}_{axis}.jpg')

        #Plot the risk on of the test
        plt.figure()
        plt.plot(risksTest)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Risk")
        plt.savefig(f'validation_risk{augmented}_{axis}.jpg')


def main():
    d = data.Data(PLAYLIST, FILECOUNT, 1000)
    X_train, y_train, X_test, y_test = d.getData()

    pr = PolynomialRegression(X_train, y_train, X_test, y_test)
    #pr.closedForm()

    pr.setGradientParams(100, 20, 0.001, 1)
    pr.gradientDescent()


if __name__ == "__main__":
    main()