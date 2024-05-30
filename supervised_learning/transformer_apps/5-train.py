import tensorflow as tf
import numpy as np
import time

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dm, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)  # Convert step to float
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    # Load and preprocess the dataset
    dataset = Dataset(batch_size, max_len)
    
    # Create the learning rate scheduler and optimizer
    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    # Initialize the transformer model
    transformer = Transformer(N, dm, h, hidden, 
                              dataset.tokenizer_pt.vocab_size + 2, 
                              dataset.tokenizer_en.vocab_size + 2, 
                              max_len, max_len)
    
    # Define loss and accuracy metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    
    # Define the training step function
    @tf.function
    def train_step(inputs, targets):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inputs, targets)
        
        with tf.GradientTape() as tape:
            predictions = transformer(inputs, targets, True, enc_padding_mask, combined_mask, dec_padding_mask)
            loss = loss_function(targets, predictions)
        
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        
        train_loss(loss)
        train_accuracy(accuracy_function(targets, predictions))
    
    # Training loop
    for epoch in range(epochs):
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for (batch, (inputs, targets)) in enumerate(dataset.data_train):
            train_step(inputs, targets)
            
            if batch % 50 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch}: Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}')
        
        print(f'Epoch {epoch + 1}: Loss {train_loss.result():.4f}, Accuracy {train_accuracy.result():.4f}')
        # print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
    
    return transformer
