import numpy as np

from LyricEmbedding import LyricEmbedding
from utils import get_verses_in_folder, tokenize_string
import re
import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense

tf.debugging.set_log_device_placement(True)


# Define general parameters for experiment
batch_size = 1  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = ["./TravisScott", "./AsapRocky", "./Drake", "./DrDre", "./JayZ", "./KanyeWest", "./KendrickLamar", "./KidCudi", "./MacMiller", "./PushaT", "./RickRoss"]

### LIST OF TODOS FOR TURNING INTO RAP LYRICS GENERATOR
# TODO: Easier to do better pre-processing than to ensure same split and strip everywhere?
# TODO: Should prob do a lstrip instead of a strip so that we keep spaces between words.

# The texts (verses) that will be input and output for model
input_texts = []
target_texts = []
# The words that will be used as input and output dicts (vectors)
input_words = set()
target_words = set()
# Get the verses you want to train on
verses = get_verses_in_folder(data_path)
# For each verse,
for verse in verses[: min(num_samples, len(verses) - 1)]:  # To the min of num_samples and file length
    # Make sure that the verse actually has some length before tokenizing it, etc. (TODO: Should fix preprocessing so that empty verses is not retrieved from get_verses_in_folder)
    if len(verse) > 1:
        # Input text is tokenized verse split on comma
        input_text = ", ".join(tokenize_string(verse))
        # Target text is whole verse. Add start and end token to target text
        target_text = "<S>" + verse + "<E>"
        input_texts.append(input_text)
        target_texts.append(target_text)
        # Add the unique words from inputs and targets to each their set.
        for word in input_text.split(","):
            if word not in input_words:
                input_words.add(word.strip())
        for word in re.split(' |\n', target_text):
            if word not in target_words:
                target_words.add(word.strip())

# Sort input and target words alphabetically
input_words = sorted(list(input_words))
target_words = sorted(list(target_words))
# Get number of encoder tokens (words) for input and target (length of word-dict basically)
num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)
# Get the max seq length for encoder and decoder (i.e. max number of words in a verse).
max_encoder_seq_length = max([len(txt.split(",")) for txt in input_texts])
max_decoder_seq_length = max([len(re.split(' |\n', txt)) for txt in target_texts])

# Print the information so far
print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

# Create dictionaries that map words to numbers for both input and target
input_token_index = dict([(word, i) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i) for i, word in enumerate(target_words)])

# Input data to encoder is one matrix for each verse - where each matrix has "max_encoder_seq_length" possible words (rows)
# and each column corresponds to each word (TODO: Double check that this logic makes sense by debugging encoder_input_data before fit)
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)


# For each of the pairs of "input_text" (tokenized words) and "target_text" (full verses), create input data for encoder
# and target data for decoder
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # Fill 1's in input_data for verse (i) for each spot (t) for each word (input_token_index[word.strip()])
    for t, word in enumerate(input_text.split(",")):
        encoder_input_data[i, t, input_token_index[word.strip()]] = 1.0
    # TODO: Line below won't work because im stripping on insert. So no existing key for " ", do i need to fix?
        # Thinking they used this here to add " " after each word (they are adding letters by single)
        # I do not need this since im inserting words in the first place(?).
    # encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0

    # Same as for input_data, but for target text
    for t, word in enumerate(re.split(' |\n', target_text)):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[word.strip()]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word.strip()]] = 1.0
    # decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
    # decoder_target_data[i, t:, target_token_index[" "]] = 1.0

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)



##### INFERENCE #####

# This is building the same model as above in a slightly complex way (?), seems like they are doing like this to skip
# initialization of num_encoder_tokens again (but since i am doing everything in same document, i should be able to do it
# in a cleaner way)!.

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_word_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_word_index = dict((i, word) for word, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["<S>"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_token_index]
        decoded_sentence += sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_word == "<E>" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence


for seq_index in range(20):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index:seq_index+1]
    input_seq2 = encoder_input_data[seq_index]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)


