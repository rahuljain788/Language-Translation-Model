# from vectorize import train_ds
# from positional_excoding import PositionalEmbedding
import tensorflow as tf
# vocab_size_en = 10000
# seq_length = 20
#
# # test the dataset
# for inputs, targets in train_ds.take(1):
#     print(inputs["encoder_inputs"])
#     embed_en = PositionalEmbedding(seq_length, vocab_size_en, embed_dim=512)
#     en_emb = embed_en(inputs["encoder_inputs"])
#     print(en_emb.shape)
#     print(en_emb._keras_mask)