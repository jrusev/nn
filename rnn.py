# Recurrent neural network that learns to add 8-bit binary numbers. At each
# time step the network adds two bits, and remebers when there is a carry.
import numpy as np
np.random.seed(1234)

MAX_BITS = 8
MAX_NUM = pow(2, MAX_BITS)

learn_rate = 0.1
input_dim, hidden_dim, output_dim = 2, 10, 1

w_0 = np.random.randn(input_dim, hidden_dim)
w_1 = np.random.randn(hidden_dim, output_dim)
w_h = np.random.randn(hidden_dim, hidden_dim)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(y):
    return y*(1-y)

# Generate a tuple (a, b, a+b). Values are binary encoded as array of 0s and 1s.
def generate_example():
    a = np.random.randint(MAX_NUM/2)
    b = np.random.randint(MAX_NUM/2)
    return [np.unpackbits(np.uint8(x)) for x in (a,b,a+b)]

def accuracy(w_0, w_1, w_h, num_examples=100):
    correct_predictions = []
    for j in range(num_examples):
        a, b, c = generate_example()
        _, l2_values = forward(w_0, w_1, w_h, a, b)
        prediction = [np.round(l2[0][0]) for l2 in l2_values[::-1]]
        x, y, z = 0, 0, 0
        for i in range(MAX_BITS):
            x +=a[::-1][i]*pow(2, i)
            y +=b[::-1][i]*pow(2, i)
            z +=prediction[::-1][i]*pow(2, i)
        print str(x) + " + " + str(y) + " = " + str(z)
        correct_predictions.append(np.array_equal(prediction,c))

    return np.mean(correct_predictions)

def forward(w_0, w_1, w_h, a, b):
    l1_values = list()
    l2_values = list()
    l1_values.append(np.zeros(hidden_dim))

    # Feed the numbers starting from the least significant bit.
    for i in range(MAX_BITS, 0, -1):
        X = np.array([[a[i-1],b[i-1]]])

        # hidden layer = input + prev_hidden
        l1 = sigmoid(np.dot(X,w_0) + np.dot(l1_values[-1],w_h))
        l2 = sigmoid(np.dot(l1,w_1))
        l1_values.append(l1)
        l2_values.append(l2)

    return l1_values, l2_values

def grads(w_0, w_1, w_h, a, b, c):
    l1_values, l2_values = forward(w_0, w_1, w_h, a, b)
    future_l1_delta = np.zeros(hidden_dim)

    w_0_grad, w_1_grad, w_h_grad = [np.zeros_like(w) for w in (w_0, w_1, w_h)]
    for i in range(MAX_BITS):
        X = np.array([[a[i],b[i]]])
        y = np.array([[c[i]]]).T
        l1 = l1_values[-i-1]
        prev_l1 = l1_values[-i-2]

        l2_delta = y - l2_values[-i-1]
        l1_delta = (future_l1_delta.dot(w_h.T) + l2_delta.dot(w_1.T)) * d_sigmoid(l1)

        w_1_grad += np.atleast_2d(l1).T.dot(l2_delta)
        w_h_grad += np.atleast_2d(prev_l1).T.dot(l1_delta)
        w_0_grad += X.T.dot(l1_delta)

        future_l1_delta = l1_delta

    return w_0_grad, w_1_grad, w_h_grad

for j in range(2000):
    a, b, c = generate_example()
    w_0_grad, w_1_grad, w_h_grad = grads(w_0, w_1, w_h, a, b, c)
    w_0 += w_0_grad * learn_rate
    w_1 += w_1_grad * learn_rate
    w_h += w_h_grad * learn_rate

    if(j % 100 == 0):
        print j, accuracy(w_0, w_1, w_h)
