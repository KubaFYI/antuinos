import numpy as np
import time
from numba import jit, njit, prange

class RNN():
    '''
    An implementation of a simple one-layer RNN.
    '''

    def __init__(self, max_agents_no, inputs_no, outputs_no, hidden_layer_size, context_layer_size=None):
        self.max_agents_no = max_agents_no
        self.inputs_no = inputs_no
        self.outputs_no = outputs_no
        self.hidden_layer_size = hidden_layer_size
        if context_layer_size is None:
            self.context_layer_size = self.hidden_layer_size
        else:
            raise NotImplementedError
            self.context_layer_size = context_layer_size

        self.idx_input_start = 0
        self.idx_input_stop = self.inputs_no
        self.idx_context_start = self.idx_input_stop
        self.idx_context_stop = self.idx_context_start + self.context_layer_size
        self.idx_output_start = self.idx_context_stop
        self.idx_output_stop = self.idx_output_start + self.outputs_no

        self.weights = np.random.random((max_agents_no,
                                         self.hidden_layer_size,
                                         self.inputs_no * self.context_layer_size * self.outputs_no))

        self.context_layer = np.zeros((max_agents_no, self.context_layer_size))

    @staticmethod
    @njit(cache=True)
    def softmax(x):
        '''
        Compute softmax values for each sets of scores in x.
        '''
        e_x = np.empty_like(x)
        for i in prange(x.shape[0]):
            e_x[i, :] = np.exp(x[i, :])
            e_x[i, :] /= np.sum(e_x[i, :])
        return e_x

    def compute(self, inputs, outputs):
        '''
        Compute the results of the RNN computation
        '''
        hidden_layer = np.empty((inputs.shape[0], self.hidden_layer_size))
        outputs[...] *= 0

        for idx in range(inputs.shape[0]):
            hidden_layer[idx, :] = np.tanh(np.dot(self.weights[idx, :, self.idx_input_start:self.idx_input_stop], inputs[idx, :]) +
                                    np.dot(self.weights[idx, :, self.idx_context_start:self.idx_context_stop], self.context_layer[idx, :]))
            # import pdb; pdb.set_trace()
            outputs[idx, :] = np.dot(self.weights[idx, :, self.idx_output_start:self.idx_output_stop].T, hidden_layer[idx, :])

        outputs[...] = RNN.softmax(outputs)
        self.context_layer = hidden_layer.copy()

        return outputs[...]
        

if __name__ == '__main__':
    q=np.random.rand(250 * 7).reshape(250, 7)
    w=np.empty((q.shape[0], 7))
    rnn=RNN(250, 7, 7, 10)
    N = 100
    start_time = time.time()
    for i in range(N):
        rnn.compute(q, w)
    print((time.time() - start_time) / N * 1000)