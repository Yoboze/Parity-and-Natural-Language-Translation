#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import urllib

##############################################################################
# Suppress warnings - use with caution!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##############################################################################
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                 key_dim=embed_dim)
        self.ffn = keras.Sequential(
                [keras.layers.Dense(ff_dim, activation="gelu"),
                keras.layers.Dense(embed_dim),])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

##############################################################################
class MaskedTransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(MaskedTransformerBlock, self).__init__()
        self.att1 = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                  key_dim=embed_dim)
        self.att2 = keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                  key_dim=embed_dim)
        self.ffn = keras.Sequential(
           [keras.layers.Dense(ff_dim, activation="gelu"),
            keras.layers.Dense(embed_dim),])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.
        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],0)
        return tf.tile(mask, mult)

    def call(self, inputs, training):
        input_shape = tf.shape(inputs[0])
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        mask = self.causal_attention_mask(batch_size,seq_len, seq_len,tf.bool)
        attn_output1 = self.att1(inputs[0], inputs[0],attention_mask = mask)
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(inputs[0] + attn_output1)
        attn_output2 = self.att2(out1, inputs[1])
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm1(out1 + attn_output2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output)

##############################################################################
class MaskedTokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(MaskedTokenAndPositionEmbedding, self).__init__()
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True)
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen+1,
        output_dim=embed_dim,
        mask_zero=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=1, limit=maxlen+1, delta=1)
        positions = positions * tf.cast(tf.sign(x),tf.int32)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

##############################################################################
class PositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2] # x already embedded
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions
##############################################################################
# Custom masked loss/accuracy functions
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

# import dataset

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


tf.config.set_visible_devices(tf.config.get_visible_devices()[0:1])
strategy = tf.distribute.OneDeviceStrategy('gpu:0')
# Size of the gestalt, context representations...
latent_dim = 60
embed_dim = 100 # Embedding size for each token
num_heads = 4 # Number of attention heads
ff_dim = 16 # Hidden layer size in feed forward network inside transformer
stack = 4
with strategy.scope():
    x = keras.layers.Input(shape=encX.shape[1:])
    y = x
    y = MaskedTokenAndPositionEmbedding(maxlen=encX.shape[1],vocab_size=len(i_to_c_eng),embed_dim=embed_dim)(y)

# Linear projection to latent_dim
    y = keras.layers.Dense(latent_dim)(y)
    for _ in range(stack):
        y = TransformerBlock(embed_dim=y.shape[2],num_heads=num_heads,ff_dim=ff_dim)(y)
    encoder = keras.Model(x,y)

with strategy.scope():
    ## Decoder Construction
    x = keras.layers.Input(shape=decX.shape[1:])
    c = keras.layers.Input(shape=encoder.output_shape[1:])
    y = x
    y = MaskedTokenAndPositionEmbedding(maxlen=decY.shape[1],vocab_size=len(i_to_c_por),embed_dim=embed_dim)(y)
    y = keras.layers.Dense(latent_dim)(y)
    for _ in range(stack):
        y = MaskedTransformerBlock(embed_dim=y.shape[2],num_heads=num_heads,ff_dim=ff_dim)([y,c])
    y = keras.layers.Dense(len(i_to_c_por),activation=keras.activations.softmax)(y)
    decoder = keras.Model([x, c],y)

    
with strategy.scope():
    x_enc = keras.layers.Input(encoder.input_shape[1:])
    x_dec = keras.layers.Input(decoder.input_shape[0][1:])
    y = decoder([x_dec,encoder(x_enc)])
    coupled = keras.Model([x_enc,x_dec],y)
# Final model...
with strategy.scope():
    coupled.compile(loss=MaskedSparseCategoricalCrossentropy,optimizer=keras.optimizers.Nadam(),metrics=[MaskedSparseCategoricalAccuracy])

# coupled.summary()
epochs = 200

with strategy.scope():
    history = coupled.fit([encX,decX], decY,epochs=epochs,validation_split=0.2,verbose=1)

    
with strategy.scope():
    preNTF = decX[-2000:].copy()
    context = encoder.predict(encX[-2000:])
    for i in range(decX[-2000:].shape[1]-1):
        postNTF = decoder.predict([preNTF,context])
        preNTF[:,i+1] = postNTF[:,i,:].argmax(-1)
    predictions = postNTF
    print(float(MaskedSparseCategoricalAccuracy(decY[-2000:],predictions)))