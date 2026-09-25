"""
Deep Learning Model Architecture for Intent Classification.
Utilizes a Multi-Layer Perceptron (MLP) with Dropout regularization
implemented via Keras / TensorFlow.
"""

import keras
from keras import layers, models


def build_intent_model(input_dim: int, num_classes: int) -> models.Sequential:
    """
    Constructs a Deep Neural Network for multi-class intent classification.

    Architecture:
    - Input: Bag-of-Words feature vector (dimension = vocabulary size)
    - Dense Layer 1: 128 units with ReLU activation
    - Dropout Layer 1: 0.5 dropout rate for regularization
    - Dense Layer 2: 64 units with ReLU activation
    - Dropout Layer 2: 0.5 dropout rate
    - Dense Output Layer: num_classes units with Softmax activation
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
