
import json
import char_cnn_model

import numpy as np
import data_helpers
np.random.seed(123)  # for reproducibility

# set parameters:

subset = None

# Whether to save model parameters
save = False
model_name_path = 'data/crepe_model.json'
model_weights_path = 'data/crepe_model_weights.h5'

# Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 100

# Model params
# Filters for conv layers
nb_filter = [256, 512, 512, 1024]
# Conv layer kernel size
filter_kernels = [7, 5, 3, 2]
# Number of units in the dense layer
dense_outputs = 2048

# Compile/fit params
batch_size = 80
nb_epoch = 200

print('Loading data...')
# Expect x to be a list of sentences. Y to be a one-hot encoding of the categories.
(xt, yt), (x_test, y_test), label_array = data_helpers.load_ag_data()
# Number of units in the final output layer. Number of classes.
cat_output = len(label_array)  # 1007

print('Creating vocab...')
vocab, reverse_vocab, vocab_size, alphabet = data_helpers.create_vocab_set()

xt = data_helpers.encode_data(xt, maxlen, vocab)
x_test = data_helpers.encode_data(x_test, maxlen, vocab)

print('Chars vocab: {}'.format(alphabet))
print('Chars vocab size: {}'.format(vocab_size))
print('X_train.shape: {}'.format(xt.shape))

print('Build model...')

model = char_cnn_model.model(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter, cat_output)

print('Fit model...')
model.summary()

model.fit(xt, yt,
          validation_data=(x_test, y_test), batch_size=batch_size, epochs=nb_epoch, shuffle=True)

if save:
    print('Saving model params...')
    json_string = model.to_json()
    with open(model_name_path, 'w') as f:
        json.dump(json_string, f)

    model.save_weights(model_weights_path)
