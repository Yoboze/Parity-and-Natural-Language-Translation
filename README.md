**Neural Network Approaches to Parity and Natural Language Translation**

This project explores the application of sequence-to-sequence neural architectures — SimpleRNN, LSTM, and Transformer — to two distinct tasks:

- **Parity prediction** (symbolic sequence modeling)

- **English–Portuguese translation** (natural language processing)

🚀 Overview
The goal was to evaluate how well different neural architectures capture dependencies in sequences — both artificial (parity) and natural language. The models were trained and tested on:

- Synthetic binary strings for parity learning

- Real-world English–Portuguese sentence pairs from a parallel corpus

🧠 Architectures Implemented
- **SimpleRNN:** Basic recurrent model with limited ability to handle long-term dependencies.

- **LSTM:** Recurrent model designed to better preserve long-range context via gating mechanisms.

- **Transformer:** Attention-based architecture capable of capturing complex relationships without recurrence.

📊 Results Summary
| Task                  | SimpleRNN Accuracy | LSTM Accuracy | Transformer Accuracy |
| --------------------- | ------------------ | ------------- | -------------------- |
| **Parity Prediction** | \~57%              | \~90%         | **\~99.7%**          |
| **Translation (NLP)** | \~24%              | \~33%         | **\~41.2%**          |

The **Transformer** consistently outperformed both SimpleRNN and LSTM across tasks, highlighting its effectiveness in capturing both symbolic logic and natural language structure.

🔧 Technical Details
- Framework: TensorFlow / Keras

- Data:

     - Parity: Generated binary sequences of varying lengths.

     - Translation: [English–Portuguese parallel dataset](https://raw.githubusercontent.com/luisroque/deep-learning-articles/main/data/eng-por.txt)


- Custom Loss & Accuracy Functions: Masked sparse categorical loss and accuracy were implemented to ignore padding tokens during training.

- Training:
  
      -All models trained for 200–1000 epochs using the Nadam optimizer.
      -Validation split used to monitor generalization performance.


✅ Key Achievements
- Successfully demonstrated the superiority of Transformer models in both symbolic and language-based sequence learning.

- Validated that attention mechanisms enable better performance in tasks involving complex sequential logic and contextual meaning.

- Applied NLP techniques to real-world translation data, showing measurable improvement over traditional recurrent models.
