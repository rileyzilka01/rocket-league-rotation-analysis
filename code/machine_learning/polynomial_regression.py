
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
        Perform closed form linear regression
        '''

        #Bias
        self.file = 'polynomial_regression.txt'
        self.xData()
        self.yData()
        self.zData()

        #Augment the features to absorb the bias term
        self.file = 'polynomial_regression_augmented.txt'
        self.augmentX()
        self.augmentY()
        self.augmentZ()
        self.xData()
        self.yData()
        self.zData()
        
    
    def xData(self):
        '''
        Get the weights and statistics for the x axis data and write it to file
        '''
        w = self.getWeights(self.X_train_x, self.y_train_x)

        #t for training, v for validation
        tMSE, tRMSE, tMAE = self.getStatistics(self.X_train_x, self.y_train_x, w)
        vMSE, vRMSE, vMAE = self.getStatistics(self.X_test_x, self.y_test_x, w)

        self.writeStats(tMSE, tRMSE, tMAE, vMSE, vRMSE, vMAE, 'w', 'X')


    def yData(self):
        '''
        Get the weights and statistics for the y axis data and write it to file
        '''
        w = self.getWeights(self.X_train_y, self.y_train_y)

        #t for training, v for validation
        tMSE, tRMSE, tMAE = self.getStatistics(self.X_train_y, self.y_train_y, w)
        vMSE, vRMSE, vMAE = self.getStatistics(self.X_test_y, self.y_test_y, w)

        self.writeStats(tMSE, tRMSE, tMAE, vMSE, vRMSE, vMAE, 'a', 'Y')


    def zData(self):
        '''
        Get the weights and statistics for the x axis data and write it to file
        '''
        w = self.getWeights(self.X_train_z, self.y_train_z)

        #t for training, v for validation
        tMSE, tRMSE, tMAE = self.getStatistics(self.X_train_y, self.y_train_y, w)
        vMSE, vRMSE, vMAE = self.getStatistics(self.X_test_y, self.y_test_y, w)

        self.writeStats(tMSE, tRMSE, tMAE, vMSE, vRMSE, vMAE, 'a', 'Z')


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


    def augmentX(self):
        '''
        Augment the x data
        '''
        self.X_train_x = np.concatenate(
            (np.ones([self.X_train_x.shape[0], 1]), self.X_train_x), axis=1)
        self.X_test_x = np.concatenate(
            (np.ones([self.X_test_x.shape[0], 1]),  self.X_test_x), axis=1)


    def augmentY(self):
        '''
        Augment the y data
        '''
        self.X_train_y = np.concatenate(
            (np.ones([self.X_train_y.shape[0], 1]), self.X_train_y), axis=1)
        self.X_test_y = np.concatenate(
            (np.ones([self.X_test_y.shape[0], 1]),  self.X_test_y), axis=1)


    def augmentZ(self):
        '''
        Augment the z data
        '''
        self.X_train_z = np.concatenate(
            (np.ones([self.X_train_z.shape[0], 1]), self.X_train_z), axis=1)
        self.X_test_z = np.concatenate(
            (np.ones([self.X_test_z.shape[0], 1]),  self.X_test_z), axis=1)


def main():
    d = data.Data(PLAYLIST, FILECOUNT)
    X_train, y_train, X_test, y_test = d.getData()

    pr = PolynomialRegression(X_train, y_train, X_test, y_test)
    pr.closedForm()


if __name__ == "__main__":
    main()