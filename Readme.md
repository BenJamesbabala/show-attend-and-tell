# show_attend_and_tell

Implemetation of [*K. Xu, Show, attend, and Tell*](http://arxiv.org/abs/1502.03044) with tensorflow

*Not Completed Yet*



##reference
author's source code with Theano : https://github.com/kelvinxu/arctic-captions

source code with Tensorflow : https://github.com/jazzsaxmafia/show_attend_and_tell.tensorflow



##code
train.py : main codes for training and evaluating performance

model.py : Caption_model, main model with Word embedding, RNN with soft-attention, decoding

model_lstm_basic.py : Caption_model_LSTM, Child Class of Caption_model

model_gru_basic.py : Caption_model_GRU, Child Class of Caption_model

bleu.py : codes for evaluating bleu score *TO FIX*
