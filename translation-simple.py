#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import urllib
url = "https://raw.githubusercontent.com/luisroque/deep-learning-articles/main/data/eng-por.txt"

def grabdata(url):
	data1 = []
	with urllib.request.urlopen(url) as raw_data:
		for line in raw_data:
			data1.append(line.decode("utf-8").split('\t')[0:2])
	data1 = np.array(data1)
	data1 =data1[0:10000]
	shuffle = np.random.permutation(data1.shape[0])
	data= data1[shuffle,:]
	return data
data =grabdata(url)
 
i_to_c_eng = [''] + list({char for word in data[:,0] for char in word})
c_to_i_eng = {i_to_c_eng[i]:i for i in range(len(i_to_c_eng))}
i_to_c_por = [''] + list({char for word in data[:,1] for char in word})
i_to_c_por.append('<START>') # Start
i_to_c_por.append('<STOP>') # Stop
c_to_i_por = {i_to_c_por[i]:i for i in range(len(i_to_c_por))}
###


max_eng = np.max([len(x) for x in data[:,0]])
max_por = np.max([len(x) for x in data[:,1]])
encX = np.array([[c_to_i_eng[c] for c in word]+[0]*(max_eng-len(word)) for word in data[:,0]])
Y = np.array([[c_to_i_por['<START>']]+
			  [c_to_i_por[c] for c in word]+[c_to_i_por['<STOP>']]+
			  [0]*(max_por+1-len(word)) for word in data[:,1]])
decX = Y[:,:-1]
decY = Y[:,1:]
encX_1h = keras.utils.to_categorical(encX,np.max(encX)+1)
decX_1h = keras.utils.to_categorical(decX,np.max(decX)+1)
decY_1h = keras.utils.to_categorical(decY,np.max(decY)+1)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
def MaskedSparseCategoricalCrossentropy(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)
	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask
	return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
def MaskedSparseCategoricalAccuracy(real, pred):
	accuracies = tf.equal(tf.cast(real,tf.int64), tf.argmax(pred, axis=2))
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	accuracies = tf.math.logical_and(mask, accuracies)
	accuracies = tf.cast(accuracies, dtype=tf.float32)
	mask = tf.cast(mask, dtype=tf.float32)
	return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


latent_dim = 100
embed_dim = 100

# Encoder
x = keras.layers.Input(shape=encX.shape[1:])
y = x
y = keras.layers.Embedding(input_dim=np.max(encX)+1,
						   output_dim = embed_dim,
						   mask_zero=True)(y)

y = keras.layers.SimpleRNN(latent_dim,activation='tanh',
						   return_sequences=False,
						   return_state=False)(y)
encoder = keras.Model(x,y,name='Encoder')

# Decoder
x1 = keras.layers.Input(shape=decX.shape[1:])
x2 = keras.layers.Input(shape=encoder.output.shape[1:])
x = [x1, x2]
y1 = keras.layers.Embedding(input_dim=np.max(decX)+1,
							output_dim = embed_dim,
							mask_zero=True)(x1)
y2 = keras.layers.RepeatVector(decX.shape[1])(x2)
y = keras.layers.Concatenate(axis=2)([y1,y2])
y = keras.layers.SimpleRNN(latent_dim,activation='tanh',
						   return_sequences=True,
						   return_state=False)(y)
y = keras.layers.Dense(decY_1h.shape[-1],
					   activation=keras.activations.softmax)(y)
decoder = keras.Model(x,y,name='Decoder')

# Coupled model...
enc_x = keras.layers.Input(shape=encX.shape[1:])
dec_x = keras.layers.Input(shape=decX.shape[1:])
x = [enc_x, dec_x]
y = encoder(enc_x)
y = decoder([dec_x,y])
coupled = keras.Model(x,y)
# Final model...
coupled.compile(loss=MaskedSparseCategoricalCrossentropy,
				optimizer=keras.optimizers.Nadam(),
				metrics=[MaskedSparseCategoricalAccuracy])

epochs = 200
history = coupled.fit([encX,decX], decY, validation_split=0.2,epochs=epochs,verbose=0)

##Use 20% for validation
decX1 =decX[-2000:]
encX1 =encX[-2000:]
decY1 =decY[-2000:]

# Without teacher forcing
preNTF = decX1.copy()
context = encoder.predict(encX1)
for i in range(decX1.shape[1]-1):
	postNTF = decoder.predict([preNTF,context])
	preNTF[:,i+1] = postNTF.argmax(-1)[:,i]
predictions = postNTF
print(decY1[-1])
print(predictions.argmax(-1)[-1])
Output = MaskedSparseCategoricalAccuracy(decY1,predictions)
print(Output.numpy())
