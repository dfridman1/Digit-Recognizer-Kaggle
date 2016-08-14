import numpy as np



class Classifier(object):
    def _normalize_labels(self, y):
        '''
           Returns a new set of labels (mapped to integers starting from zero).
           Also computes a dictionary from converting from new labels to original ones.

           Input: y - an N-dimensional array comprising labels.
           Returns: labels mapped to { 0, 1, ..., |y.unique()|-1 }
        '''
        self.from_index_label_to_raw = dict((i, l) for i, l in enumerate(np.unique(y)))
        from_label_to_index = dict((l, i)
                                   for (i, l) in self.from_index_label_to_raw.items())

        return np.vectorize(lambda l: from_label_to_index[l])(y)


    def to_label(self, index):
        '''
           Maps an index (output by 'predict' method) to 'raw' label.

           Input: index - an integer.
           Returns: label - an integer.
        '''
        return self.from_index_label_to_raw[index]
