# Embedding lookdown

[![Packagist](https://img.shields.io/badge/Tensorflow-1.3.0-yellow.svg)]()
[![Packagist](https://img.shields.io/badge/Tensorlayer-1.6.4-blue.svg)]()


Abstrace
---
In traditional seq2seq model, the predict words can be translate automatically by the model. In latest tensorflow version, the `train_helper` class is added to generate the corresponding tag. However, there's few mechanism that can transfer the word embedding vector into index sequence. This repository is the extension of tensorlayer. The `EmbeddingReverseLayer` is defined in this project.

How to lookdown
---
1. The layer will compute the **L2 distance** between whole word vectors and input embedding
2. The distance will add negative sign and shift over zero
3. The softmax will shrink the vector as probability view
4. The index with maximum probability will be picked    

Usage
---
1. Import the `layer_extern.py` file.
2. Use `EmbeddingReverseLayer` layer, for example:
```python
>>> # network is the graph which had contained embedding
>>> network = layer_extern.EmbeddingReverseLayer(network, 'embedding', 'embedding_reverse')
```
You can check `demo.py` for more detail.    
<br/>


Notice
---
1. You should define the embedding layer first, then use `EmbeddingReverseLayer`.
2. The parameter - embedding name in `EmbeddingReverseLayer` should correspond to the name of embedding layer
*  Since this project didn't limit you that you can only define 1 word embedding, this design can allow you to define several word embedding in single graph.    