# 🌟 Autoencoders 🌟

## 📝 Description 
Autoencoders are a type of neural network that are used to learn efficient codings of unlabeled data. The network is trained to compress the input into a lower-dimensional code and then reconstruct the output from this representation. This project focuses on implementing various types of autoencoders, including vanilla, sparse, convolutional, and variational autoencoders. Each model aims to capture the most important features of the input data and reconstruct the original input as accurately as possible using the learned features.

The project demonstrates practical applications of autoencoders in reducing dimensionality and feature learning, making it an essential part of machine learning and deep learning explorations. The models are implemented using TensorFlow and are designed to handle different types of input data, showcasing the versatility and power of autoencoders in learning generative models and deep representations.

## 📚 Ressources
- [Autoencoder - definition](https://intranet.hbtn.io/rltoken/WoU4g9-ukc3CprULNz7bFQ)
- [Autoencoder - loss function](https://intranet.hbtn.io/rltoken/mwNwl6NAjS5Bq5NgOrkAIw)
- [Deep learning - deep autoencoder](https://intranet.hbtn.io/rltoken/i-6H-NSF1EeTS-wO9XJOKw)
- [Introduction to autoencoders](https://intranet.hbtn.io/rltoken/KOQVo5SHrT9KQs8DR-77XQ)
- [Variational Autoencoders - EXPLAINED!](https://intranet.hbtn.io/rltoken/GOnvaZYR8DKYpzhNgHST8w) up to 12:55
- [Variational Autoencoders](https://intranet.hbtn.io/rltoken/nmXflFRZVSepLyIeWj7ZZw)
- [Intuitively Understanding Variational Autoencoders](https://intranet.hbtn.io/rltoken/oz7kL8wcgYA_L_f7rl_ITw)
- [Deep Generative Models](https://intranet.hbtn.io/rltoken/8NsQZhtwAbFodpIT94zQlA) up to Generative Adversarial Networks

## 🛠️ Technologies and Tools Used
- **Python**: Chosen for its versatility and widespread use in machine learning. It supports numerous libraries that facilitate neural network implementation.
- **TensorFlow**: Utilized for building and training the autoencoder models due to its robustness and efficiency in handling large datasets.
- **NumPy**: Essential for handling large arrays and matrices, which are crucial in data manipulation for neural networks.

## 📋 Prerequisites
- **Python Version**: 3.8
- **TensorFlow Version**: 2.6
- **NumPy Version**: 1.19.2
- All scripts should be run on Ubuntu 20.04 LTS, ensuring compatibility and performance.

## 🚀 Installation and Configuration
To get started with the Autoencoders project, follow these step-by-step instructions:

1. **Clone the Repository**: Start by cloning the repo to your local machine using Git:

```sh
git clone https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/autoencoders
```

2. **Set Up Your Environment**: Ensure that you are running Ubuntu 20.04 LTS and have Python 3.8 installed. It is recommended to use a virtual environment:

```sh
python -m venv env
source env/bin/activate
```

3. **Install Required Libraries**: Install the necessary Python libraries specified in the prerequisites:

```sh
pip install tensorflow==2.6 numpy==1.19.2
```

4. **Navigate to the Project Directory**: After installation, navigate to the project directory:

```sh
cd holbertonschool-machine_learning/unsupervised_learning/autoencoders
```
These steps will set up the necessary environment and dependencies to begin working with the autoencoder models.

## 💡 Usage
To utilize the autoencoder models, you can run the provided Python scripts. Here are some examples of how to execute and interact with the models:

- **Running the Vanilla Autoencoder**:

```sh
python3 0-vanilla.py
```
This script will train the vanilla autoencoder model and display the input and reconstructed images.

- **Evaluating the Sparse Autoencoder**:

```sh
python3 1-sparse.py
```
Run this to train the sparse autoencoder and visualize the sparsity in the encoded representations.

## ✨ Key Features

1. **Vanilla Autoencoder**: Basic form of an autoencoder without any regularization. Great for understanding the fundamental mechanism of autoencoders.
2. **Sparse Autoencoder**: Implements L1 regularization to induce sparsity in the latent representations, enhancing feature selection.
3. **Convolutional Autoencoder**: Utilizes convolutional layers to better handle image data, making it suitable for image compression tasks.
4. **Variational Autoencoder**: This model not only learns to encode but also to generate new data that resembles the input data, useful for generative tasks.

## 📝 List of Tasks
Below is a list of tasks associated with the Autoencoders project, detailing the key components and objectives of each task:

| Number | Task | Description |
| ------ | ---------------------- | ------------------------------------------------------------------------------- |
| 0 | [Vanilla Autoencoder](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/autoencoders#vanilla-autoencoder) | Create and train a basic autoencoder model to understand the foundational principles. |
| 1 | [Sparse Autoencoder](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/autoencoders#sparse-autoencoder) | Implement sparsity in the autoencoder using L1 regularization to enhance feature representation. |
| 2 | [Convolutional Autoencoder](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/autoencoders#convolutional-autoencoder) | Use convolutional layers to process image data more effectively, ideal for tasks like image compression. |
| 3 | [Variational Autoencoder](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/autoencoders#variational-autoencoder) | Develop an autoencoder that learns a probabilistic latent space, useful for generating new data instances. |

Each task is linked to its respective section in the GitHub repository, where more detailed instructions and code can be found.

## 📬 Contact
For any further inquiries or if you would like to contribute to the project, please feel free to reach out through the following contact link:

- **LinkedIn Profile**: [Caroline CHOCHOY](https://www.linkedin.com/in/caroline-chochoy62/)
