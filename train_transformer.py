import matplotlib.pyplot as plt
import tensorflow as tf

# the dataset objects from Lesson 03
from vectorize import train_ds, val_ds
# the building block functions from Lesson 08
from build_tensformer import transformer, CustomSchedule, masked_loss, masked_accuracy


# Create and train the model
seq_len = 20
num_layers = 4
num_heads = 8
key_dim = 128
ff_dim = 512
dropout = 0.1
vocab_size_en = 10000
vocab_size_fr = 20000
model = transformer(num_layers, num_heads, seq_len, key_dim, ff_dim,
                    vocab_size_en, vocab_size_fr, dropout)
lr = CustomSchedule(key_dim)
optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
epochs = 20
history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

# Save the trained model
model.save("eng-fra-transformer.h5")

# Plot the loss and accuracy history
# fig, axs = plt.subplots(2, figsize=(6, 8), sharex=True)
# fig.suptitle('Traininig history')
# x = list(range(1, epochs+1))
# axs[0].plot(x, history.history["loss"], alpha=0.5, label="loss")
# axs[0].plot(x, history.history["val_loss"], alpha=0.5, label="val_loss")
# axs[0].set_ylabel("Loss")
# axs[0].legend(loc="upper right")
# axs[1].plot(x, history.history["masked_accuracy"], alpha=0.5, label="acc")
# axs[1].plot(x, history.history["val_masked_accuracy"], alpha=0.5, label="val_acc")
# axs[1].set_ylabel("Accuracy")
# axs[1].set_xlabel("epoch")
# axs[1].legend(loc="lower right")
# plt.show()