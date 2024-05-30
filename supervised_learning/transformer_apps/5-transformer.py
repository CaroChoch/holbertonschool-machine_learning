#!/usr/bin/env python3
""" Transformer Network """
import tensorflow as tf
import numpy as np


def positional_encoding(max_seq_len, dm):
    """ calculates the positional encoding for a transformer
    Arguments:
        - max_seq_len is an integer representing the maximum sequence length
        - dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing
        the positional encoding vectors
    """
    # Initialize the positional encoding matrix with zeros
    PE = np.zeros((max_seq_len, dm))

    # Loop over each position in the sequence
    for i in range(max_seq_len):
        # Loop over each dimension of the positional encoding
        for j in range(0, dm, 2):
            # Compute the positional encoding using sin (even indices)
            PE[i, j] = np.sin(i / (10000 ** (j / dm)))
            # Compute the positional encoding using cos (odd indices)
            PE[i, j + 1] = np.cos(i / (10000 ** (j / dm)))

    # Return the positional encoding matrix
    return PE

def sdp_attention(Q, K, V, mask=None):
    """
    Scaled Dot Product Attention
    Arguments:
        - Q: tensor with shape (..., seq_len_q, dk) containing the query matrix
        - K: tensor with shape (..., seq_len_v, dk) containing the key matrix
        - V: tensor with shape (..., seq_len_v, dv) containing the value matrix
        - mask: tensor that can be broadcast into (..., seq_len_q, seq_len_v)
            containing the optional mask, or defaulted to None
    Returns: output, weights
        - output: tensor with shape (..., seq_len_q, dv) containing the scaled
            dot product attention
        - weights: tensor with shape (..., seq_len_q, seq_len_v) containing
            the attention weights
    """
    # Perform the dot product between Q (queries) and K (keys) and transpose K
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Get the dimensionality of the keys
    keys_dimentionality = tf.cast(tf.shape(K)[-1], tf.float32)

    # Scale the dot product by the square root of dimensionality of the keys
    scaled_attention_logits = matmul_qk / tf.math.sqrt(keys_dimentionality)

    # Apply the mask if provided
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Apply softmax to get the attention weights
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Multiply the weights by the values (V)
    output = tf.matmul(weights, V)

    # Return the attention output and the attention weights
    return output, weights

class MultiHeadAttention(tf.keras.layers.Layer):
    """ Multi Head Attention """

    def __init__(self, dm, h):
        """
        Class constructor
        Arguments:
            - dm is an integer representing the dimensionality of the model
            - h is an integer representing the number of heads
            - dm is diVisible by h
                * h - the number of heads
                * dm - the dimensionality of the model
                * depth - the depth of each attention head
                * WQ - a Dense layer with dm units, used to generate the
                    Query matrix
                * WK - a Dense layer with dm units, used to generate the
                    Key matrix
                * WV - a Dense layer with dm units, used to generate the
                    Value matrix
                * linear - a Dense layer with dm units, used to generate
                    the attention output
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        # Dense layer used to generate the Query matrix
        self.Wq = tf.keras.layers.Dense(dm)
        # Dense layer used to generate the Key matrix
        self.Wk = tf.keras.layers.Dense(dm)
        # Dense layer used to generate the Value matrix
        self.Wv = tf.keras.layers.Dense(dm)

        # Dense layer used to generate the attention output
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension
        Arguments:
            - x is a tensor of shape (batch, seQ_len, dm)
                containing the input to split
            - batch_size is an integer representing the batch size
        Returns: a tensor with shape (batch, h, seQ_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Call Method
        Arguments:
            - Q is a tensor of shape (batch, seQ_len_Q, dm)
                containing the input to generate the Query matrix
            - K is a tensor of shape (batch, seQ_len_V, dm)
                containing the input to generate the Key matrix
            - V is a tensor of shape (batch, seQ_len_V, dm)
                containing the input to generate the Value matrix
            - masK is always None
        Returns: output, weights
            - output a tensor with its last two dimensions as (...,
                seQ_len_Q, dm)
                containing the scaled dot product attention
            - weights a tensor with its last three dimensions as
                (..., h, seQ_len_Q, seQ_len_V) containing the attention weights
        """
        batch_size = tf.shape(Q)[0]

        # Generate the Query, Key, and Value matrices
        Q = self.Wq(Q)  # (batch, seq_len_q, dm)
        K = self.Wk(K)  # (batch, seq_len_v, dm)
        V = self.Wv(V)  # (batch, seq_len_v, dm)

        # Split and transpose for multi-head attention
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled Dot Product Attention
        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        # Transpose and reshape the scaled_attention back
        # to (batch, seq_len_q, dm)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention,
            (batch_size, -1, self.dm))

        # Final linear layer
        output = self.linear(concat_attention)  # (batch, seq_len_q, dm)

        return output, weights

class EncoderBlock(tf.keras.layers.Layer):
    """ Class EncoderBlock """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        Arguments:
            - dm: Dimensionality of the model
            - h: Number of heads
            - hidden: Number of hidden units in the fully connected layer
            - drop_rate: Dropout rate
        Parameters:
            - mha: MultiHeadAttention layer
            - dense_hidden: Hidden dense layer with hidden units and relu
                activation
            - dense_output: Output dense layer with dm units
            - layernorm1: First layer normalization layer, with epsilon=1e-6
            - layernorm2: Second layer normalization layer, with epsilon=1e-6
            - dropout1: First dropout layer
            - dropout2: Second dropout layer
        """
        super(EncoderBlock, self).__init__()

        # Multi-head attention
        self.mha = MultiHeadAttention(dm, h)
        # Hidden layer
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # Output layer
        self.dense_output = tf.keras.layers.Dense(dm)
        # Layer Normalization 1
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Layer Normalization 2
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Dropout 1
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        # Dropout 2
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Method to call the instance
        Arguments:
            - x: Tensor of shape (batch, input_seq_len, dm) containing
                the input to the encoder block
            - training: Boolean to determine if the model is training
            - mask: Mask to be applied for multi-head attention
        Returns: A tensor of shape (batch, input_seq_len, dm) containing
            the block’s output
        """
        # Multi-head attention
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        # Add and Norm
        normed_attention_output = self.layernorm1(x + attention_output)
        # Feed forward
        feedforward_output = self.dense_hidden(normed_attention_output)
        feedforward_output = self.dense_output(feedforward_output)
        feedforward_output = self.dropout2(
            feedforward_output, training=training)
        # Add and Norm
        encoder_output = self.layernorm2(
            normed_attention_output + feedforward_output)

        return encoder_output

class DecoderBlock(tf.keras.layers.Layer):
    """ Class DecoderBlock """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor
        Arguments:
            - dm: Dimensionality of the model
            - h: Number of heads
            - hidden: Number of hidden units in the fully connected layer
            - drop_rate: Dropout rate
        Parameters:
            - mha1: First MultiHeadAttention layer (Masked Multi-Head
                Attention)
            - mha2: Second MultiHeadAttention layer (Encoder-Decoder Attention)
            - dense_hidden: Hidden dense layer with hidden units and
                relu activation
            - dense_output: Output dense layer with dm units
            - layernorm1: First layer normalization layer, with epsilon=1e-6
            - layernorm2: Second layer normalization layer, with epsilon=1e-6
            - layernorm3: Third layer normalization layer, with epsilon=1e-6
            - dropout1: First dropout layer
            - dropout2: Second dropout layer
            - dropout3: Third dropout layer
        """
        super(DecoderBlock, self).__init__()

        # First MultiHeadAttention layer (Masked Multi-Head Attention)
        self.mha1 = MultiHeadAttention(dm, h)
        # Second MultiHeadAttention layer (Encoder-Decoder Attention)
        self.mha2 = MultiHeadAttention(dm, h)
        # Hidden layer
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # Output layer
        self.dense_output = tf.keras.layers.Dense(dm)
        # Layer Normalization 1
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Layer Normalization 2
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Layer Normalization 3
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Dropout 1
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        # Dropout 2
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        # Dropout 3
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Method to call the instance
        Arguments:
            - x: Tensor of shape (batch, target_seq_len, dm) containing the
                input to the decoder block
            - encoder_output: Tensor of shape (batch, input_seq_len, dm)
                containing the output of the encoder
            - training: Boolean to determine if the model is training
            - look_ahead_mask: Mask to be applied to the first multi-head
                attention layer
            - padding_mask: Mask to be applied to the second multi-head
                attention layer
        Returns: A tensor of shape (batch, target_seq_len, dm) containing the
            block’s output
        """
        # First multi-head attention block (Masked Multi-Head Attention)
        masked_attention_output, _ = self.mha1(x, x, x, look_ahead_mask)
        masked_attention_output = self.dropout1(
            masked_attention_output,
            training=training)
        normed_masked_attention_output = self.layernorm1(
            x + masked_attention_output)

        # Second multi-head attention block (Encoder-Decoder Attention)
        enc_dec_attention_output, _ = self.mha2(
            normed_masked_attention_output,
            encoder_output,
            encoder_output,
            padding_mask)
        enc_dec_attention_output = self.dropout2(
            enc_dec_attention_output, training=training)
        normed_enc_dec_attention_output = self.layernorm2(
            normed_masked_attention_output + enc_dec_attention_output)

        # Feed forward neural network
        feed_forward_neural_output = self.dense_hidden(
            normed_enc_dec_attention_output)
        feed_forward_neural_output = self.dense_output(
            feed_forward_neural_output)
        feed_forward_neural_output = self.dropout3(
            feed_forward_neural_output,
            training=training)
        decoder_block_output = self.layernorm3(
            normed_enc_dec_attention_output + feed_forward_neural_output)

        return decoder_block_output

class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder
    """
    def __init__(self, N, dm, h, hidden, input_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Initialize the Transformer Encoder
        Arguments:
            - N: Number of blocks in the encoder
            - dm: Dimensionality of the model
            - h: Number of heads in the multi-head attention mechanism
            - hidden: Number of hidden units in the fully connected layer
            - input_vocab: Integer representing the size of the input
                vocabulary
            - max_seq_len: Integer representing the maximum sequence length
            - drop_rate: Dropout rate
        """
        super(Encoder, self).__init__()
        self.dm = dm
        self.N = N
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = []
        for _ in range(N):
            self.blocks.append(EncoderBlock(dm, h, hidden, drop_rate))
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Call method for the Transformer Encoder
        Arguments:
            - x: Tensor of shape (batch, input_seq_len) containing the input
                to the encoder
            - training: Boolean to determine if the model is in training mode
            - mask: Mask to be applied for multi-head attention
        Returns:
            - Tensor of shape (batch, input_seq_len, dm) containing the encoder
                output
        """
        seq_len = x.shape[1]
        x = self.embedding(x)  # Apply embedding
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))  # Scale embeddings
        x += self.positional_encoding[:seq_len]  # Add positional encoding
        x = self.dropout(x, training=training)  # Apply dropout

        for i in range(self.N):  # Apply each encoder block
            x = self.blocks[i](x, training, mask)

        # Return the encoder output
        return x

class Decoder(tf.keras.layers.Layer):
    """ Decoder class """
    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """
        Class constructor
        Arguments:
            - N: Number of blocks in the decoder
            - dm: Dimensionality of the model
            - h: Number of heads in the multi-head attention mechanism
            - hidden: Number of hidden units in the fully connected layer
            - target_vocab: Size of the target vocabulary
            - max_seq_len: Maximum sequence length possible
            - drop_rate: Dropout rate
        """
        super(Decoder, self).__init__()
        self.dm = dm
        self.N = N
        # Embedding layer for target vocabulary
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        # Positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # Create and append each decoder block
        self.blocks = []
        for _ in range(N):
            self.blocks.append(DecoderBlock(dm, h, hidden, drop_rate))
        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Method to call the decoder
        Arguments:
            - x: Tensor of shape (batch_size, target_seq_len, dm) containing
                the input to the decoder
            - encoder_output: Tensor of shape (batch_size, input_seq_len, dm)
                containing the output of the encoder
            - training: Boolean to determine if the model is training
            - look_ahead_mask: Mask to be applied to the first multi-head
                attention layer
            - padding_mask: Mask to be applied to the second multi-head
                attention layer
        Returns:
            - Tensor of shape (batch_size, target_seq_len, dm) containing the
                decoder output
        """
        seq_len = x.shape[1]
        x = self.embedding(x)  # Apply embedding to input
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))  # Scale embeddings
        x += self.positional_encoding[:seq_len]  # Add positional encoding
        x = self.dropout(x, training=training)  # Apply dropout

        for i in range(self.N):  # Apply each decoder block
            x = self.blocks[i](
                x,
                encoder_output,
                training,
                look_ahead_mask,
                padding_mask)

        # Return the decoder output
        return x

class Transformer(tf.keras.layers.Layer):
    """ Transformer class """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor
        Arguments:
            - N: Number of blocks in the encoder and decoder
            - dm: Dimensionality of the model
            - h: Number of heads
            - hidden: Number of hidden units in the fully connected layers
            - input_vocab: Size of the input vocabulary
            - target_vocab: Size of the target vocabulary
            - max_seq_input: Maximum sequence length possible for the input
            - max_seq_target: Maximum sequence length possible for the target
            - drop_rate: Dropout rate
        """
        super(Transformer, self).__init__()
        # Encoder layer
        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate)
        # Decoder layer
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate)
        # Final linear layer
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """
        Method to call the transformer
        Arguments:
            - inputs: Tensor of shape (batch, input_seq_len) containing
                the inputs
            - target: Tensor of shape (batch, target_seq_len) containing
                the target
            - training: Boolean to determine if the model is training
            - encoder_mask: Padding mask to be applied to the encoder
            - look_ahead_mask: Mask to be applied to the decoder
            - decoder_mask: Padding mask to be applied to the decoder
        Returns:
            - Tensor of shape (batch, target_seq_len, target_vocab) containing
                the transformer output
        """
        # Encoder output
        enc_output = self.encoder(inputs, training, encoder_mask)
        # Decoder output
        dec_output = self.decoder(
            target,
            enc_output,
            training,
            look_ahead_mask,
            decoder_mask)
        # Linear layer
        final_output = self.linear(dec_output)
        # Return final output
        return final_output
