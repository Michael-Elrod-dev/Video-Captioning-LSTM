# Video-Captioning-LSTM

This project implements a video captioning system using an LSTM-based encoder-decoder architecture with an attention mechanism. The model takes video features as input and generates textual descriptions (captions) for the videos.

## Project Structure

- `model.py`: Contains the neural network architecture (Encoder, Decoder, Attention mechanism).
- `train.py`: Handles data processing, model training, and evaluation.
- `test.py`: Performs inference on test data and calculates BLEU scores.
- `bleu_eval.py`: Implements BLEU score calculation for evaluating caption quality.

## Features

- LSTM-based encoder-decoder architecture
- Attention mechanism for improved caption generation
- Teacher forcing with adaptive ratio during training
- BLEU score evaluation

