# Height-to-Weight Prediction using Keras

This project implements a simple linear regression model using a neural network with Keras to predict weight based on height. The dataset contains height and weight data for individuals, and the model is trained to find the relationship between these two variables.

## Overview

The neural network in this project consists of a single neuron (unit) with a linear activation function. It is designed to perform a simple regression task, learning a linear relationship between the height (input) and the weight (output).

### Model Architecture

1. **Input**: Height (1 feature)
2. **Output**: Predicted weight (1 output)
3. **Layer**: 
   - A single fully connected (Dense) layer with 1 neuron.
   - Activation function: `linear`
   - Weight and bias initialized to zeros.

### Dataset

- The dataset (`weight-height.csv`) contains two columns:
  - **Height**: Height of the individual (input feature)
  - **Weight**: Weight of the individual (target variable)
  
The data is used to train the model in predicting the weight based on the height.

### Code

```python
import pandas
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
dataset = pandas.read_csv('weight-height.csv')
y = dataset['Weight']
X = dataset['Height']

# Build the model
brain = Sequential()
brain.add(Dense(units=1, activation="linear", bias_initializer="zeros", kernel_initializer="zeros"))

# Compile the model with mean absolute error as the loss function
brain.compile(loss='mean_absolute_error')

# Train the model for 10 epochs
brain.fit(X, y, epochs=10)

# View the model weights and make predictions
brain.get_weights()
brain.predict(X)
