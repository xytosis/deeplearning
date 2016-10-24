import re
import tensorflow as tf
import numpy as np
import sys

sess = tf.InteractiveSession()

# initializes the weights with some noise
def make_weight(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

# initializes the biases with some noise
def make_bias(shape):
  return tf.Variable(tf.constant(0.1, shape = shape))

def tokenizer(sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
    token_count = dict()
    token_count["STOP"] = 0
    words = []
    for space_separated_fragment in sentence.strip().lower().split():
        fragments = re.split(word_split, space_separated_fragment)
        for frag in fragments:
            words.append(frag)
            if frag in token_count:
                token_count[frag] += 1
            else:
                token_count[frag] = 1
            if frag == "." or frag == "!" or frag == "?":
                words.append("STOP")
                token_count["STOP"] += 1
    del token_count[""]
    real_tokens = dict(sorted(token_count.items(), key=lambda x: x[1], reverse=True)[:8000])
    to_return = []
    for w in words:
        if w in real_tokens:
            to_return.append(w)
        else:
            to_return.append("*UNK*")
    return to_return

#########################################################################################

with open(sys.argv[1], "r") as f:
    a = f.read()
    tokens = tokenizer(a)
    vocab_size = len(set(tokens))

vocab = list(set(tokens))

word_to_id = dict()
for i in range(0, vocab_size):
    word_to_id[vocab[i]] = i

num_steps = 20
h_size = 256
batch_size = 20
chunk_size = num_steps * batch_size
epochs = 1

training_temp = tokens[:len(tokens) * 9 / 10]
testing_temp = tokens[len(tokens) * 9 / 10:]

training = map(lambda x: word_to_id[x], training_temp)
testing = map(lambda x: word_to_id[x], testing_temp)

training_x = []
training_y = []
testing_x = []
testing_y = []

for i in range(len(training) / chunk_size):
    start = i * chunk_size
    temp_x = []
    temp_y = []
    for j in range(batch_size):
        temp_x.append(training[start + j * num_steps:start + (j + 1) * num_steps])
        temp_y.append(training[start + j * num_steps + 1:start + (j + 1) * num_steps + 1])
    training_x.append(temp_x)
    training_y.append(temp_y)

for i in range(len(testing) / chunk_size):
    start = i * chunk_size
    temp_x = []
    temp_y = []
    for j in range(batch_size):
        temp_x.append(testing[start + j * num_steps:start + (j + 1) * num_steps])
        temp_y.append(testing[start + j * num_steps + 1:start + (j + 1) * num_steps + 1])
    testing_x.append(temp_x)
    testing_y.append(temp_y)

print len(training_x)
print len(testing_x)

#########################################################################################
    
inpt = tf.placeholder(tf.int32, [None, None])
output = tf.placeholder(tf.int32, [None, None])
keep_prob = tf.placeholder(tf.float32)

E = tf.Variable(tf.truncated_normal([vocab_size, 50], stddev = 0.1))

W = make_weight([h_size, vocab_size])
b = make_bias([vocab_size])

blstm = tf.nn.rnn_cell.BasicLSTMCell(h_size)

initial_state = blstm.zero_state(batch_size, tf.float32)

embed = tf.nn.embedding_lookup(E, inpt)

# drop out some stuff
dropout = tf.nn.dropout(embed, keep_prob)

rnn_output, final_state = tf.nn.dynamic_rnn(blstm, dropout, initial_state=initial_state)

h1_reshape = tf.reshape(rnn_output, [chunk_size, h_size])

logits = tf.matmul(h1_reshape, W) + b

error = tf.reduce_sum(tf.nn.seq2seq.sequence_loss_by_example([logits], [tf.reshape(output, [-1])], [tf.ones([chunk_size])])) / chunk_size

train_step = tf.train.AdamOptimizer(0.0001).minimize(error)

sess.run(tf.initialize_all_variables())
total_error = 0.0
print "TRAINING"
for epoch in range(epochs):
    print "EPOCH " + str(epoch)
    state = initial_state.eval()
    for i in range(len(training_x)):
        loss, state, _ = sess.run([error, final_state, train_step],
            feed_dict={inpt: training_x[i], output: training_y[i], initial_state: state, keep_prob: 0.5})
        print loss, i

    state = initial_state.eval()
    total_error = 0.0
    print "TESTING"
    for i in range(len(testing_x)):
        loss, state = sess.run([error, final_state],
            feed_dict={inpt: testing_x[i], output: testing_y[i], initial_state: state, keep_prob: 1.0})
        total_error += loss
        print loss, i, total_error, "testing"

print np.exp(total_error/len(testing_x))