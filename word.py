# import tensorflow as tf


# inpt = tf.placeholder(tf.float32, [None])
# E = tf.variable(tf.random_uniform([vocabsize, embedsize], -1.0, 10))
# embed = tf.nn.embedding_lookup(E, inpt)

# W1
# b1

# W2
# b2

# h1 = tf.relu(tf.matmul(embd, W1) + b1) # first hidden computation
# h2 = tf.matmul(h1, W2) + b2 # logits, and we do the softmax of this

# error = tf.nn.sparse_softmax_cross_entropy_with_logits(h2, labels) # label is the unique id for each word
# # error is -log(p(c))
# # preplexity = e^error

import tensorflow as tf
import numpy as np

# initializes the weights with some noise
def make_weight(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

# initializes the biases with some noise
def make_bias(shape):
  return tf.Variable(tf.constant(0.1, shape = shape))

sess = tf.InteractiveSession()
#sess.run(tf.initialize_all_variables())

s = set()
all_words = []

with open("train.txt", "r") as f:
    for line in f:
        s.update(line.split())
        all_words.extend(line.split())

vocab = sorted(list(s))

vocab_size = len(vocab)

word_to_id = dict()
for i in range(0, vocab_size):
    word_to_id[vocab[i]] = i

# one_hots = tf.one_hot(range(0, vocab_size), vocab_size)


# here we initialize the flow graph
inpt = tf.placeholder(tf.int32, [None])
output = tf.placeholder(tf.int32, [None])

E = tf.Variable(tf.truncated_normal([vocab_size, 30], stddev = 0.1))

embed = tf.nn.embedding_lookup(E, inpt)

W1 = make_weight([30, 100])
b1 = make_bias([100])

W2 = make_weight([100, vocab_size])
b2 = make_bias([vocab_size])

h1 = tf.nn.relu(tf.matmul(embed, W1) + b1)

h2 = tf.matmul(h1, W2) + b2

error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h2, output))
train_step = tf.train.AdamOptimizer(0.01).minimize(error)
x = 0
batch_size = 20
sess.run(tf.initialize_all_variables())
while x < len(all_words) - 1:
    input_batch_words = all_words[x:x + batch_size]
    word_ids_inpt = map(lambda x: word_to_id[x], input_batch_words)
    output_batch_words = all_words[x + 1: x + 1 + batch_size]
    word_ids_output = map(lambda x: word_to_id[x], output_batch_words)

    x += batch_size
    _, err = sess.run([train_step, error], feed_dict = {inpt: word_ids_inpt, output: word_ids_output})

all_test_words = []

# now we do testing
with open("test.txt") as f:
    for line in f:
        all_test_words.extend(line.split())

test_input_batch = all_test_words[:-1]
test_input_ids = map(lambda x: word_to_id[x], test_input_batch)
test_output_batch = all_test_words[1:]
test_output_ids = map(lambda x: word_to_id[x], test_output_batch)

err = sess.run([error], feed_dict= {inpt : test_input_ids, output: test_output_ids})
print np.exp(err)