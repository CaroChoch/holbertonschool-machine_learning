# 🌟 Time Series Forecasting with RNNs 🌟

## 📝 Description 
Time Series Forecasting is a challenging task that involves making predictions based on historical data points indexed in time order. In this project, the focus is on using Recurrent Neural Networks (RNNs) to forecast Bitcoin (BTC) prices. The goal is to predict the closing value of BTC after a given time window using the previous 24-hour data. The project leverages deep learning to solve a real-world financial problem.

This project demonstrates how to preprocess time series data, create a data pipeline in TensorFlow, and build effective RNN models for forecasting. It tackles the complexity of financial time series data, which often contains noise, and highlights the advantages of using RNNs for such tasks. The solution employs a robust machine learning approach to handle raw BTC data and deliver accurate forecasts.

## 📚 Resources

- [Time Series Prediction](https://intranet.hbtn.io/rltoken/IPHyUFv7WJ_cxTINQmtgNg)
- [Time Series Forecasting](https://intranet.hbtn.io/rltoken/nIebldUW1xMYyP604lPmtA)
- [Time Series Talk: Stationarity](https://intranet.hbtn.io/rltoken/Z6gxuejq_ftwd64E-UPAYg)
- [tf.data: Build TensorFlow input pipelines](https://intranet.hbtn.io/rltoken/qwJQkXozMMU3FEi9upt-qw)
- [Tensorflow Datasets](https://intranet.hbtn.io/rltoken/rQ78XqULPk8Ad6D__cnRWw)
- [Time Series](https://intranet.hbtn.io/rltoken/nA9DP5YSKunSZOdceKp6pQ)
- [Stationary Process](https://intranet.hbtn.io/rltoken/FcSWzi08147D5eQioh-lbw)
- [tf.keras.layers.SimpleRNN](https://intranet.hbtn.io/rltoken/0b-RuS_OStuKKP9vZspm3g)
- [tf.keras.layers.GRU](https://intranet.hbtn.io/rltoken/SK2e7ZrNhZxC-vh-BeOhUQ)
- [tf.keras.layers.LSTM](https://intranet.hbtn.io/rltoken/s6BR0PAxk1Ygn3eVTGm6pQ)
- [tf.data.Dataset](https://intranet.hbtn.io/rltoken/jRwxnbX7t2AvVDcnVJqg0w)

## 🛠️ Technologies and Tools Used

- Python: The main programming language used for building and training the model.
- TensorFlow: The deep learning framework used for creating RNN models.
- NumPy: A fundamental package for scientific computing with Python.
- Pandas: A library providing high-level data structures for data analysis.
- tf.data.Dataset: TensorFlow's utility for building efficient input pipelines.

## 📋 Prerequisites

- ![Python](https://img.shields.io/badge/python-3.5-blue)
- ![NumPy](https://img.shields.io/badge/numpy-1.15-green)
- ![TensorFlow](https://img.shields.io/badge/tensorflow-2.4.1-orange)
- ![Pandas](https://img.shields.io/badge/pandas-1.1.5-red)

## 📊 Data Files

<details>
<summary>forecast_btc.py</summary>
<br>
#!/usr/bin/env python3
'''
Forecasting BTC prices using RNNs
'''

import tensorflow as tf
import pandas as pd

# Your code implementation here
</details>

<details>
<summary>preprocess_data.py</summary>
<br>
#!/usr/bin/env python3
'''
Preprocessing BTC data
'''

import pandas as pd
import numpy as np

# Your code implementation here
</details>

## 🚀 Installation and Configuration

1. **Step 1: Clone the Repository**

```sh
git clone https://github.com/CaroChoch/holbertonschool-machine_learning.git
```

2. **Step 2: Navigate to the Project Directory**

```sh
cd holbertonschool-machine_learning/supervised_learning/time_series
```
3. **Step 3: Install the Requirements**

```sh
pip install -r requirements.txt
```

4. **Step 4: Run the Preprocessing Script**

```sh
./preprocess_data.py
```

5. **Step 5: Run the Forecasting Script**

```sh
./forecast_btc.py
```

## 💡 Usage

- **Example 1: Run the Model**

```sh
./forecast_btc.py
```
This will start the training and validation of the model for BTC price forecasting.

- **Exemple 2: Preprocess the Data**
```sh
./preprocess_data.py
```
This script preprocesses the raw BTC data, preparing it for the forecasting model.

## ✨ Main Features

1. BTC Price Forecasting: This project uses Recurrent Neural Networks (RNNs) to forecast Bitcoin (BTC) prices based on historical data.
2. Data Preprocessing: It includes a script for preprocessing raw BTC data, including filtering, rescaling, and preparing the data for the model.
3. TensorFlow Input Pipeline: The project demonstrates how to create efficient input pipelines for time series data using TensorFlow's tf.data.Dataset.
4. RNN Architectures: The project showcases different RNN architectures, such as SimpleRNN, GRU, and LSTM, for time series forecasting.
5. Performance Evaluation: The model's performance is evaluated using Mean Squared Error (MSE) as the cost function, providing insights into the accuracy of the forecasts.

## 📝 List of Tasks

| Number | Task                                          | Description                                                        |
| ------ | --------------------------------------------- | ------------------------------------------------------------------ |
| 0      | [When to Invest](https://github.com/CaroChoch/holbertonschool-machine_learning/blob/main/supervised_learning/time_series/forecast_btc.py) | Create, train, and validate an RNN model for forecasting BTC prices.  |
| 1      | [Everyone wants to know](https://www.linkedin.com/in/caroline-chochoy62/) | Write a blog post explaining the process of completing the task.  |

## 📬 Contact

- LinkedIn Profile: [Caroline Chochoy](https://www.linkedin.com/in/caroline-chochoy62/)
