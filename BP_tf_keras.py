import numpy as np
import tensorflow as tf

N_data = 100000
x_data = np.float32(10*np.random.rand(N_data,2))
y_data = (np.dot(x_data**2, [2, 1]) + 3).reshape((N_data,1))

x_test_data = np.float32(10*np.random.rand(10,2))
y_test_data = (np.dot(x_test_data**2, [2, 1]) + 3).reshape((10,1))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='sigmoid',input_dim=2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer = tf.train.GradientDescentOptimizer(0.001),
                loss = 'mse',
                metrics = ['mae']
)

model.fit(x_data, y_data, epochs=10, batch_size=32)

result_predict = model.predict(x_test_data, batch_size = 32)
print(np.concatenate((y_test_data,result_predict), axis=1))
