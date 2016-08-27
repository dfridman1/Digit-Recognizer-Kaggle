import numpy as np

from classifier import Classifier




class NeuralNetwork(Classifier):

    '''3-layer Neural Network with ReLU activation function'''

    def __init__(self, input_size, layer_1_size, layer_2_size, number_of_classes):
        self.D = input_size
        self.H1 = layer_1_size
        self.H2 = layer_2_size
        self.C = number_of_classes

        self.params = {}
        self.params['W1'] = np.random.randn(self.D, self.H1) * np.sqrt(2./self.D)
        self.params['W2'] = np.random.randn(self.H1, self.H2) * np.sqrt(2./self.H1)
        self.params['W3'] = np.random.randn(self.H2, self.C) * np.sqrt(2./self.H2)
        self.params['b1'] = np.ones(self.H1) * 0.01
        self.params['b2'] = np.ones(self.H2) * 0.01
        self.params['b3'] = np.ones(self.C) * 0.01


    def copy(self):
        nn = NeuralNetwork(self.D, self.H1, self.H2, self.C)
        return nn


    def train(self,
              X,
              y,
              learning_rate=1e-7,
              learning_rate_decay=0.98,
              reg=1e-3,
              num_iters=500,
              batch_size=-1,
              n=100,
              verbose=False):

        y = self._normalize_labels(y)

        N, D = X.shape

        loss_history = []
        for it in xrange(1, num_iters + 1):

            if batch_size != -1:
                indices = np.random.choice(N, size=batch_size, replace=True)
                X_batch = X[indices]
                y_batch = y[indices]
            else:
                X_batch = X
                y_batch = y

            loss, grads = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            for param_name in self.params:
                self.params[param_name] -= learning_rate * grads[param_name]

            if verbose and it % 1000 == 0:
                print 'iteration: %d, loss: %f' % (it, loss)

            if it % n == 0:
                learning_rate *= learning_rate_decay

        return np.array(loss_history)


    def predict(self, X):
        scores = self.predict_scores(X)

        y_pred = np.argmax( scores, axis=1 )
        return np.vectorize(self.to_label)(y_pred)


    def predict_scores(self, X):
        W1, b1, W2, b2, W3, b3 = (self.params['W1'],
                                  self.params['b1'],
                                  self.params['W2'],
                                  self.params['b2'],
                                  self.params['W3'],
                                  self.params['b3'])
        scores = np.maximum(0,
                            np.maximum(0,
                                       X.dot(W1) + b1).dot(W2) + b2).dot(W3) + b3
        return scores


    def loss(self, X, y, reg=0):

        N, _ = X.shape

        W1, b1, W2, b2, W3, b3 = (self.params['W1'],
                                  self.params['b1'],
                                  self.params['W2'],
                                  self.params['b2'],
                                  self.params['W3'],
                                  self.params['b3'])

        # computing score
        
        h1_scores = X.dot(W1) + b1
        h1_relu   = np.maximum(0, h1_scores)

        h2_scores = h1_relu.dot(W2) + b2
        h2_relu   = np.maximum(0, h2_scores)

        scores = h2_relu.dot(W3) + b3

        unnormalized_probs = np.exp(scores)
        normalizer = np.sum( unnormalized_probs, axis=1 ).reshape(-1, 1)
        probs = unnormalized_probs / normalizer
        correct_label_probs = probs[np.arange(N), y]

        loss = np.sum( -np.log(correct_label_probs) )
        loss /= N
        loss += 0.5 * reg * ( np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) )

        # computing gradient (backprop)

        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores /= N

        db3 = np.sum(dscores, axis=0)
        dW3 = h2_relu.T.dot(dscores)
        dW3 += reg * W3

        dh2_relu = dscores.dot(W3.T)
        dh2_scores = (h2_scores > 0).astype(float) * dh2_relu

        db2 = np.sum(dh2_scores, axis=0)
        dW2 = h1_relu.T.dot(dh2_scores)
        dW2 += reg * W2

        dh1_relu = dh2_scores.dot(W2.T)
        dh1_scores = (h1_scores > 0).astype(float) * dh1_relu

        db1 = np.sum(dh1_scores, axis=0)
        dW1 = X.T.dot(dh1_scores)
        dW1 += reg * W1

        grads = {'W1' : dW1,
                 'W2' : dW2,
                 'W3' : dW3,
                 'b1' : db1,
                 'b2' : db2,
                 'b3' : db3 }

        return loss, grads
