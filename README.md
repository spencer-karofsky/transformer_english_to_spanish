# Building a Transformer for English-to-Spanish Translation
#### Spencer Karofsky

### High-Level Description

I am very enthusiastic about AI/ML, and after several years of AI projects, I wanted to challenge myself by building a transformer.

I implemented and trained a transformer for English-to-Spanish Machine Translation based on my understanding of the [Attention is All You Need](https://arxiv.org/pdf/1706.03762) paper in PyTorch. While PyTorch already provides a powerful Transformer implementation, I chose to build my own to better understand each component and the underlying mechanics.

### Features

* Uses the [OPUS Books](https://opus.nlpl.eu/opus-100.php) dataset (English <-> Spanish)
* Implements and trains a transformer architecture inspired by the original paper based on [Attention is All You Need](https://arxiv.org/pdf/1706.03762)
* Runs inference on nine English example sentences of increasing complexity

### Model and Training Details

| Parameter                | Value                     |
| ------------------------ | ------------------------- |
| Total Parameters         | 135M                      |
| Epochs                   | 30                        |
| Training Time            | \~8 hours                 |
| Device                   | MacBook Pro M4 Max        |
| Batch Size               | 32                        |
| Sequence Length          | 128 tokens                |
| Optimizer                | Adam                      |
| Learning Rate            | 1e-4        |
| Layers (Encoder/Decoder) | 6 / 6                     |
| Attention Heads          | 8                         |
| Embedding Dimension      | 512                       |
| Hidden FFN Dimension      | 2048                       |
| Dropout                  | 0.1                       |


### Results

I evaluated this model on nine English sentences, of increasing complexity:

| English (Input)     | Spanish Sentence (Generated Output)       | Output Translated Back to English (https://www.spanishdict.com)
| ------------------------ | ------------------------- | ------- |
|  I like to eat apples and bananas.  |  me gustan las comidas y la hambre.  |   I like food and hunger.   |
|  She is reading a book in the sun.   |  leyó un libro en el sol.  |  He read a book in the sun.    |
|  They play soccer every Sunday afternoon.  |     aban los gritos de la mañana.     |    The screams of the morning were coming up.  |
|    We couldn't find the restaurant despite using the map.   |  utábamos el diván, pese a la tragedia.   |   We used the couch, despite the tragedy.   |
|  If it rains tomorrow, we'll cancel the hike.  |       che, si no llovíamos aprender la tumba.          |   Hey, if it didn't rain, we would learn the grave.   |
|  The teacher explained the problem in a different way. |     tó el profesor de camino, lo que debían modo.        |   The teacher on the way, what they should have done.   |
| Although the train was late, we still made it on time. |      , el tren nos llevaba, todavía mucho tiempo.      |   , the train was still taking us a long time.   |
| The decision, which had been debated for months, was finally announced. |  ía la decisión, que había sido anunciado por meses, fue anunciado.  |    The decision, which had been announced for months, was announced.  |
| He acted as though nothing had happened, despite knowing the consequences. |   to, no había nada, no había nada, a pesar de lo que sabía, a pesar de la consecuencia.     |   There was nothing, there was nothing, in spite of what I knew, in spite of the consequence.   |

### Reflection

This model serves as a functional proof of concept for English-to-Spanish machine translation given my hardware constraints. While the translations are often imperfect when translated back to English, the model consistently generates well-formed Spanish sentences without producing gibberish or invalid tokens—even if some of the outputs lack semantic accuracy. Overall, I am very satisfied with this project's outcome.

### Future Work

I plan to use PyTorch’s built-in Transformer implementation to train a larger and more sophisticated model, targeting a well-defined, real-world application.

I hope to train future models using a cloud service such as AWS of GCP.