# Hyperparameter Tuning



## 📝 Description
Hyperparameter Tuning is a crucial aspect of optimizing machine learning algorithms. This project involves the systematic adjustment of hyperparameters to improve the performance of models, specifically within unsupervised learning settings. The project aims to implement and refine strategies such as Gaussian Processes, Bayesian Optimization, and different search methods (grid, random) to find optimal configurations.

## 📚 Resources
- [Hyperparameter Tuning in Practice](https://intranet.hbtn.io/rltoken/USxbmvohxYUZ_oExRDy9-A)
- [Gaussian Process Regression Guide](https://intranet.hbtn.io/rltoken/LLZsTNyA3DYuxu-74h6dnA)
- [Bayesian Optimization Techniques](https://intranet.hbtn.io/rltoken/HBuXSKcATV2MdaQdGEU5hQ)
- [GPyOpt Official Documentation](https://intranet.hbtn.io/rltoken/7dtugJomWYrn83tH59J6Xg)
- [GPy Library Overview](https://intranet.hbtn.io/rltoken/m8Vwz7rK_PwUbjLBXb1Kfw)

## 🛠️ Technologies and Tools Used
- **Python**: Main programming language for scripting and algorithm implementation.
- **NumPy**: Essential library for high-performance scientific computing and data analysis.
- **GPy and GPyOpt**: Libraries used for Gaussian processes and Bayesian optimization respectively.

## 📋 Prerequisites
- Python 3.5 or above
- NumPy 1.15 or compatible version
- GPy and GPyOpt libraries installed

## 📊 Data Files
```python
# Example data used in hyperparameter tuning
X_init = np.random.uniform(-np.pi, 2*np.pi, (2, 1))
Y_init = f(X_init)
```

## 🚀 Installation and Configuration

1. Clone the repository:

```sh
git clone https://github.com/CaroChoch/holbertonschool-machine_learning.git

cd unsupervised_learning/hyperparameter_tuning
```

2. Install required Python packages:

```sh
pip install --user GPy
pip install --user gpyopt
```

## 💡 Usage

To execute hyperparameter tuning, run:

```python
python3 <script_name>.py
```

Replace `<script_name>` with the actual name of the Python script for tuning.

## ✨ Main Features

1. **Gaussian Process Initialization**: Establishes a baseline Gaussian process from initial data.
2. **Bayesian Optimization**: Utilizes Bayesian techniques to optimize hyperparameters dynamically.

## 📝 Task List

| Number | Task                                      | Description                                                             |
| ------ | ----------------------------------------- | ----------------------------------------------------------------------- |
| 0      | [**Initialize Gaussian Process**](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/hyperparameter_tuning/0-gp.py) | Initialize a 1D Gaussian process, setting the foundation for further optimization processes. |
| 1      | [**Gaussian Process Prediction**](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/hyperparameter_tuning/1-gp.py) | Update the Gaussian Process class to include prediction capabilities, essential for assessing new data points. |
| 2      | [**Update Gaussian Process**](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/hyperparameter_tuning/2-gp.py) | Enhance the Gaussian Process class to allow updates with new sample points, crucial for dynamic learning. |
| 3      | [**Initialize Bayesian Optimization**](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/hyperparameter_tuning/3-bayes_opt.py) | Set up Bayesian Optimization, configuring the class with initial parameters for optimizing the black-box function. |
| 4      | [**Bayesian Optimization - Acquisition**](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/hyperparameter_tuning/4-bayes_opt.py) | Implement the acquisition function within Bayesian Optimization to determine the next best sample location. |
| 5      | [**Optimize with Bayesian Optimization**](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/hyperparameter_tuning/5-bayes_opt.py) | Execute the optimization process, iterating to refine the hyperparameters and achieve optimal performance. |
| 6      | [**Bayesian Optimization with GPyOpt**](https://github.com/CaroChoch/holbertonschool-machine_learning/tree/main/unsupervised_learning/hyperparameter_tuning/6-bayes_opt.py) | Use GPyOpt to further streamline the Bayesian Optimization process, focusing on multiple hyperparameter adjustments. |

## 📬 Contact

LinkedIn Profile: [Caroline CHOCHOY](https://www.linkedin.com/in/caroline-chochoy62/)
