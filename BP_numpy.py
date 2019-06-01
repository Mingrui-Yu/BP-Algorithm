import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N_data = 100000
x_data = np.float32(10*np.random.rand(N_data,2)).T
y_data = (np.dot(x_data.T**2, [2, 1]) + 3).reshape((N_data,1)).T

x_test_data = np.float32(10*np.random.rand(10,2)).T
y_test_data = (np.dot(x_test_data.T**2, [2, 1]) + 3).reshape((10,1)).T

x_test_data_x = np.arange(0,10,0.1)
x_test_data_y = np.arange(0,10,0.1)
X, Y = np.meshgrid(x_test_data_x, x_test_data_y)
X_test = np.vstack((X.reshape(1,X.size), Y.reshape(1,Y.size)))

class BP_Model():
    def __init__(self):
        input_dim = 2
        n1_nodes = 128
        output_dim = 1

        self.w1 = np.random.randn(input_dim, n1_nodes).T*0.01
        self.b1 = np.zeros((n1_nodes,1))
        self.w2 = np.random.randn(n1_nodes,output_dim).T
        self.b2 = np.zeros((output_dim,1))
        self.batch_loss = []
        self.episode_loss = []

    def predict(self, x):
        z1 = np.matmul(self.w1, x) + self.b1
        a1 = self.sigmoid_fun(z1)
        z2 = np.matmul(self.w2, a1) + self.b2
        return z2

    def batch_backward(self, x, y, lr):
        m = np.shape(x)[1]

        z1 = np.matmul(self.w1, x) + self.b1
        a1 = self.sigmoid_fun(z1)
        z2 = np.matmul(self.w2, a1) + self.b2

        self.batch_loss.append(np.mean((z2-y)**2))

        dz2 = z2 - y
        dw2 = 1/m * np.matmul(dz2, a1.T)
        db2 = np.mean(dz2, axis=1, keepdims=True)

        dz1 = np.matmul(self.w2.T, dz2) * (a1*(1-a1))
        dw1 = 1/m * np.matmul(dz1, x.T)
        db1 = np.mean(dz1, axis=1, keepdims=True)

        self.w1 = self.w1 - lr*dw1
        self.b1 = self.b1 - lr*db1
        self.w2 = self.w2 - lr*dw2
        self.b2 = self.b2 - lr*db2

    def train(self, x, y, lr, batch_size, episode):
        m = np.shape(x)[1]
        for i in range(0, episode):
            for j in range(int(m/batch_size)):
                x_batch = x[:, j:j+batch_size]
                y_batch = y[:, j:j+batch_size]
                self.batch_backward(x_batch, y_batch, lr)
            self.episode_loss.append(np.mean(self.batch_loss))
            self.batch_loss = []

    def sigmoid_fun(self, x):
        return 1/(1+np.exp(-x))


if __name__ == '__main__':

    model = BP_Model()
    model.train(x_data, y_data, lr=0.001, batch_size=32, episode=10)
    
    result_predict = model.predict(X_test)
    Z = result_predict.reshape(X.shape)

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    plt.show()

    plt.plot(model.episode_loss,'-')
    print(model.episode_loss)
    plt.show()

    predict = model.predict(x_test_data)
    print(np.concatenate((y_test_data.T, predict.T), axis=1))