# Keras Tuner: Bayesian Optimization for Neural Architecture Search

This project demonstrates how to use Keras Tuner with Bayesian Optimization to find the best hyperparameters for a neural network. The model is designed for sentiment analysis on the IMDB dataset.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmassaron/keras_tuner/blob/main/keras-tuner.ipynb)

## 📝 Table of Contents
- [Keras Tuner: Bayesian Optimization for Neural Architecture Search](#keras-tuner-bayesian-optimization-for-neural-architecture-search)
  - [📝 Table of Contents](#-table-of-contents)
  - [🧐 About the Project](#-about-the-project)
  - [🚀 Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [🤖 Model Architecture](#-model-architecture)
  - [🛠️ Hyperparameter Tuning](#️-hyperparameter-tuning)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)


## 🧐 About the Project

This project provides a hands-on example of using Keras Tuner to automate the search for the optimal neural network architecture. It uses a sentiment analysis task on the IMDB dataset to showcase the power of Bayesian Optimization in hyperparameter tuning.

The `keras-tuner.ipynb` notebook contains the full code, from data preprocessing to model definition, tuning, and evaluation.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

This project requires Python 3 and the following libraries:

- TensorFlow 2.x
- Keras Tuner
- TensorFlow Addons
- scikit-learn
- Matplotlib

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lmassaron/keras_tuner.git
    cd keras_tuner
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You may need to create a `requirements.txt` file with the libraries listed above.)*

## Usage

The main code is in the `keras-tuner.ipynb` Jupyter Notebook. You can open it and run the cells sequentially to:

1.  Load and preprocess the IMDB dataset.
2.  Define the model building function with tunable hyperparameters.
3.  Instantiate the Keras Tuner with the Bayesian Optimization algorithm.
4.  Run the hyperparameter search.
5.  Retrieve the best model and evaluate its performance.

## 🤖 Model Architecture

The neural network architecture is defined in the `create_tunable_model` function. It consists of:

-   An Embedding layer to learn word representations.
-   Convolutional layers (Conv1D) to capture local patterns.
-   Recurrent layers (GRU or LSTM) to process sequential information.
-   An attention mechanism to focus on the most relevant parts of the input.
-   Dense layers for classification.

The number of layers, units, dropout rates, and other parameters are defined as hyperparameters to be tuned by Keras Tuner.

## 🛠️ Hyperparameter Tuning

Keras Tuner is used to find the best combination of hyperparameters for the model. The following tuners are available:

-   `RandomSearch`
-   `Hyperband`
-   `BayesianOptimization`

This project uses `BayesianOptimization` to efficiently search the hyperparameter space. The search space is defined in the `create_tunable_model` function using `hp.Int`, `hp.Float`, and `hp.Choice`.

## 🤝 Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.