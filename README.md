# ByteTransformer

A Transformer architecture that attempts to model text at the byte level (or any stream of bytes).
The model constructs it's own "word" boundaries **and** representations by using fixed-attention-windows in the early layers, then combining neighboring bytes in later layers.

The training task has two parts: 
1. Predict the next byte in the sequence (forwards and backwards)
2. Given two consecutive byte sequences, predict whether they are ordered correctly.
