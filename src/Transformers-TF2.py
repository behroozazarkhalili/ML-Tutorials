# %%
import tensorflow as tf
from tensorflow.keras.layers import Embedding
import numpy as np

# %%
class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    Transformer Encoder Layer Class
    """
    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        """
        Initializes the TransformerEncoderLayer.

        Parameters:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            dff (int): The number of units in the feedforward neural network layer.
            rate (float, optional): The dropout rate. Default is 0.1.

        Returns:
            None
        """
        super(TransformerEncoderLayer, self).__init__()

        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model, dropout=rate
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Call function for the layer.

        Parameters:
            inputs (tf.Tensor): The input tensor.
            training (bool): Whether the model is in training mode.

        Returns:
            tf.Tensor: The output tensor.

        """
        attn_output = self.multi_head_attention(inputs, inputs, return_attention_scores=False)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# %%
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional Encoding Class
    """
    def __init__(self, position: int, d_model: int):
        """
        Initialize the PositionalEncoding object.

        Parameters:
            position (int): The position parameter.
            d_model (int): The d_model parameter.

        Returns:
            None
        """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position: tf.Tensor, i: tf.Tensor, d_model: int) -> tf.Tensor:
        """
        Get angles for positional encoding

        Parameters:
            position (tf.Tensor): The position tensor.
            i (tf.Tensor): The i tensor.
            d_model (int): The d_model parameter.

        Returns:
            tf.Tensor: The angles tensor.
        """

        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        """
        Calculate the positional encoding.

        Parameters:
            position (int): The position parameter.
            d_model (int): The d_model parameter.

        Returns:
            tf.Tensor: The positional encoding tensor.
        """

        angle_rads = self.get_angles(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                     i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                                     d_model=d_model
                                     )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Call function for the layer
        """
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


# %%
class TransformerEncoder(tf.keras.layers.Layer):
    """
    Transformer Encoder Class.
    """
    def __init__(self, 
                 num_layers: int, 
                 d_model: int, 
                 num_heads: int, 
                 dff: int,
                 input_vocab_size: int, 
                 rate: float = 0.1):
        """
        Initializes the TransformerEncoder.

        Parameters:
            num_layers (int): The number of layers.
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of attention heads.
            dff (int): The number of neurons in the feedforward network.
            input_vocab_size (int): The size of the input vocabulary.
            rate (float, optional): The dropout rate. Defaults to 0.1.

        Returns:
            None
        """
        super(TransformerEncoder, self).__init__()

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(input_vocab_size, d_model)

        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]

    def call(self, x: tf.Tensor, training: bool) -> tf.Tensor:
        """ 
        Call function for the layer.

        Parameters:
            x (tf.Tensor): The input tensor.
            training (bool): Whether the model is in training mode.

        Returns:
            tf.Tensor: The output tensor.

        """
        
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(x.shape[-1], dtype=tf.float32))
        x += self.pos_encoding(x)
        for layer in self.enc_layers:
            x = layer(x, training)
        return x


# %%
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom Learning Rate Schedule Class

    Parameters:
        d_model (int): The dimension of the model.
        warmup_steps (int, optional): The number of warmup steps. Defaults to 4000.

    Returns:
        None
    """
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> tf.Tensor:
        """
        Call function for the learning rate schedule.

        Parameters:
            step (int): The current step.

        Returns:
            tf.Tensor: The learning rate tensor.
        """
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# %%
class Transformer(tf.keras.Model):
    """
    Transformer Class
    """
    def __init__(self, 
                 num_layers: int, 
                 d_model: int, 
                 num_heads: int, 
                 dff: int, 
                 input_vocab_size: int, 
                 rate: float = 0.1):
        """
        Initializes the Transformer model with the specified parameters.

        Parameters:
            num_layers (int): The number of layers in the Transformer model.
            d_model (int): The dimensionality of the model.
            num_heads (int): The number of attention heads.
            dff (int): The dimensionality of the feed-forward layer.
            input_vocab_size (int): The size of the input vocabulary.
            rate (float, optional): The dropout rate. Defaults to 0.1.

        Returns:
            None
        """
        
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=input_vocab_size,
            rate=rate
            )

        self.final_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(input_vocab_size, activation='softmax'))

    def call(self, inp: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Call function for the Transformer model.

        Parameters:
            inp (tf.Tensor): The input tensor.
            training (bool): The training flag.

        Returns:
            tf.Tensor: The output tensor.
        """

        enc_output = self.encoder(inp, training)
        return self.final_layer(enc_output)

# %%
def generate_data(num_samples: int, sequence_length: int, vocab_size: int) -> np.ndarray:
    """
    Function to generate random data

    Parameters:
        num_samples (int): The number of samples to generate.
        sequence_length (int): The length of the sequence.
        vocab_size (int): The size of the vocabulary.

    Returns:
        np.ndarray: The generated data.
    """
    return np.random.randint(0, vocab_size, size=(num_samples, sequence_length))


def reverse_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Function to reverse a sequence

    Parameters:
        sequence (np.ndarray): The sequence to reverse.

    Returns:
        np.ndarray: The reversed sequence.
    """
    return sequence[:, ::-1]

# %%
# Define hyperparameters
num_layers_enc = 4
d_model_enc = 128
num_heads_enc = 8
dff_enc = 512
input_vocab_size_model = 20
batch_size = 64
sequence_length = 15
epochs = 50

# Generate random training data
x_train = generate_data(1000, sequence_length, input_vocab_size_model)
y_train = reverse_sequence(x_train)[..., np.newaxis]

# Generate random test data
x_test = generate_data(1000, sequence_length, input_vocab_size_model)
y_test = reverse_sequence(x_test)[..., np.newaxis]


# %%
# Create Transformer model
transformer_model = Transformer(
    num_layers=num_layers_enc,
    d_model=d_model_enc,
    num_heads=num_heads_enc,
    dff=dff_enc,
    input_vocab_size=input_vocab_size_model
    )

# Define the optimizer with a custom learning rate schedule
custom_learning_rate = CustomSchedule(d_model_enc)
optimizer = tf.keras.optimizers.Adam(learning_rate=custom_learning_rate)

# Compile the model
transformer_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Train the model
transformer_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

# Evaluate the model
evaluation = transformer_model.evaluate(x_test, y_test)
print("Evaluation Loss:", evaluation[0])
print("Evaluation Accuracy:", evaluation[1])