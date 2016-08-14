import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import itertools
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork



print 'loading data...'
filename = 'data/train.csv'
dataset = np.array(pd.read_csv(filename))
X, y = dataset[:, 1:].astype(float), dataset[:, 0].astype(int)


print 'preprocessing...'
mu = np.mean(X, axis=0)
X -= mu


print 'splitting up...'
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42)



_, D = X.shape
layer_1_size, layer_2_size = 500, 250
number_of_classes = len(np.unique(y))
print 'number of classes = %d' % number_of_classes


best_net = None
best_loss_history = None
best_val_accuracy = -1

num_iters = 50000
batch_size = 100
n = 1000  # decay learning rate after every n iterations (at a rate .98)


learning_rates = [.01]
regularization_strengths = [.01]

print 'cross validating...'
for lr, reg in itertools.product(learning_rates, regularization_strengths):
    print 'lr: %f, reg: %f' % (lr, reg)
    net = NeuralNetwork(D, layer_1_size, layer_2_size, number_of_classes)
    loss_history = net.train(Xtr,
                             ytr,
                             learning_rate=lr,
                             reg=reg,
                             num_iters=num_iters,
                             batch_size=batch_size,
                             n=n,
                             verbose=True)
    train_accuracy = np.mean( net.predict(Xtr) == ytr )
    val_accuracy   = np.mean( net.predict(Xval) == yval )

    print 'train accuracy: %f, validation accuracy: %f' % (train_accuracy,
                                                           val_accuracy)    

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_loss_history = loss_history
        best_net = net


print
print 'accuracies for best net...'
train_accuracy = np.mean( best_net.predict(Xtr) == ytr )
val_accuracy   = np.mean( best_net.predict(Xval) == yval )
print 'train accuracy: %f, validation accuracy: %f' % (train_accuracy, val_accuracy)


print 'plotting loss history...'
plt.plot(best_loss_history)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('loss history')
plt.xscale('log')
plt.yscale('log')
plt.show()



reply = raw_input('write to file? ')
if reply in ['y', 'yes']:
    print 'loading test data...'
    filename = 'data/test.csv'
    index_col, target_col = 'ImageId', 'Label'
    df_test = pd.read_csv(filename)
    Xtest = np.array(df_test).astype(float)
    index = np.arange(1, Xtest.shape[0] + 1)

    print 'preprocessing test data...'
    Xtest -= mu

    print 'making predictions'
    y_pred = best_net.predict(Xtest)

    print 'writing to file'
    path = 'submissions/3nn_%d_%d.csv' % (layer_1_size, layer_2_size)
    df_out = pd.DataFrame()
    df_out[index_col] = index
    df_out[target_col] = y_pred
    df_out.to_csv(path, index=False)
