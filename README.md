# ChronosForge

ChronosForge is a comprehensive library bridging time-series forecasting, probabilistic modeling, and cutting-edge machine-learning techniques. Inspired by the Greek god of time, **Chronos**, this library symbolizes precision, inevitability, and the exploration of truth through data.

---

## 🌌 Philosophical Inspiration

In Greek mythology, Chronos embodies time and its infinite nature. This library reflects that spirit by crafting tools that transcend time—forecasting the future, analyzing the present, and uncovering the hidden patterns of the past. Whether solving stochastic equations, creating neural transport models, or designing probabilistic machine learning algorithms, **ChronosForge** equips you with the tools to forge solutions in the ever-flowing river of time.

---

## 🚀 Features

ChronosForge includes a variety of modules and algorithms spanning multiple disciplines:

### **Time Forecasting**
- **ARCH**: Analyze volatility and financial time series.
- **LSTM, Transformers**: Deep learning models for sequence prediction.
- **Stochastic Volatility Models**: Predict uncertainty in financial markets.
- **Causal Discovery**: Uncover causal relationships in time-series data.

### **Probabilistic Machine Learning**
- **Gaussian Processes (GP)**: Flexible models for uncertainty quantification.
- **Variational Inference (VI)**: Scalable Bayesian inference techniques.
- **Markov Models (HMM, DMM)**: For sequential data analysis.
- **Monte Carlo Methods**: No U-turn sampler to estimate the posterior distribution![CodeCogsEqn](https://github.com/user-attachments/assets/e6ca9fe8-3f34-4359-8936-459e4b0382f8)


### **Optimization and Theoretical Bounds**
- **Constrained Lagrangian Optimization**: Solve constrained optimization problems.
![CodeCogsEqn](https://github.com/user-attachments/assets/f38dd6cf-7fc1-4ffe-89b7-54a6d8ba44ae)

- **Theoretical Bounds**: Analyze generalization using PAC-Bayes, VC dimensions, and Hoeffding inequalities.

### **Deep Learning Applications**
- **Long Sequence Classification**: Handle document-level tasks with transformers and hierarchical models.
- **Temporal Fusion Transformer (TFT)**: Integrate time-dependent and contextual data.
- **Reinforcement Learning**: Optimize decision-making in dynamic environments.

### **FFT Circulant Applications**
The **FFT Circulant Modules** ChronosForge features comprehensive modules that leverage the properties of circulant matrices and the Fast Fourier Transform (FFT) to enable efficient matrix-vector multiplication, significantly reducing the number of weights in neural networks.
![CodeCogsEqn](https://github.com/user-attachments/assets/20b050c3-0207-4f0b-aed3-2f52ab55dc61)
The eigenvectors of a circulant matrix depend only on the size of the matrix, not on the elements of the matrix. Furthermore, these eigenvectors are the columns of the FFT matrix. The eigenvalues depend on the matrix entries, but the eigenvectors do not.
![CodeCogsEqn](https://github.com/user-attachments/assets/07c0c15d-5027-4899-8732-4e56a309542c)
Each element of the FFT matrix represents a complex exponential corresponding to a rotation in the frequency domain. As a result, this technique is highly effective in scenarios where the features exhibit periodicity.

![FFT Circulant Example](images/fft_circulant_example.png)

![FFT Efficienty](images/FFT_viz.png)

---

## 📊 Visualizing the Power of ChronosForge

### **1. Time-Series Sentiment Analysis**
ChronosForge provides sentiment analysis tools to evaluate market trends. Below is a 3D scatter plot of predicted sentiments for different companies.

![3D Scatter Plot of Sentiments](images/3d_scatter_plot_sentiments.png)

---

### **2. Word Clouds for Sentiment Classes**
ChronosForge visualizes sentiment-driven keywords for better interpretability. Here’s a word cloud generated for **Sentiment Class 1**.

![Word Cloud for Sentiment Class 1](images/word_cloud_sentiment_1.png)

---

### **3. Stock Price Forecasting**
ChronosForge's stochastic models can predict stock prices with high accuracy. Below is a comparison of **actual** vs. **predicted** stock prices for Novo Nordisk using ensemble techniques.

![Actual vs. Predicted Stock Prices](images/stock_price_forecasting.png)

We Provide summaries and visualizations to evaluate and interpret model performance effectively.
![Model performance across folds](images/mp.png)
---

### **4. Brownian Motion, Heston Models and Jump Diffusion**
ChronosForge incorporates advanced stochastic models to simulate and analyze financial markets with greater realism:

#### **Brownian Motion**
Brownian motion forms the backbone of stochastic processes in finance, modeling the random behavior of stock prices:

![CodeCogsEqn](https://github.com/user-attachments/assets/19ba8d69-d7c0-47e9-b9b6-c0a9b799bd68)



#### **Jump Diffusion**
Jump diffusion adds discrete jumps to Brownian motion, modeling sudden market movements:
\[
![CodeCogsEqn](https://github.com/user-attachments/assets/309650d1-a11d-4139-b3c0-6f5b2c58a9a3)
\]

> **Visualization**:

![Jump Diffusion Simulation](images/jump_diffusion_simulation.png)

---

## 📁 Directory Structure

```
ChronosForge/
├── time_forecasting/         # Modules for forecasting and time-series analysis
├── probabilistic_ml/         # Probabilistic and Bayesian methods
├── optimization/             # Optimization algorithms and theoretical bounds
├── inference/                # MCMC, variational inference, and related topics
├── neural_models/            # LSTMs, Transformers, and TCN implementations
├── structural_bioinformatics # Bioinformatics-focused tools
├── data_utils/               # Data preprocessing and augmentation tools
├── README.md                 # Library overview
└── requirements.txt          # Python dependencies
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/chronosforge.git
cd chronosforge
pip install -r requirements.txt
```

---

## 🔥 Quick Start

Here's an example of using ChronosForge for time-series forecasting with a transformer model:

```python
from chronosforge.time_forecasting.transformer import TimeSeriesTransformer
from chronosforge.utils.data_loader import load_time_series

# Load time-series data
data = load_time_series("path/to/dataset.csv")

# Initialize and train the transformer model
model = TimeSeriesTransformer(input_dim=10, hidden_dim=128, num_heads=4)
model.train(data, epochs=50, learning_rate=1e-4)

# Make predictions
predictions = model.predict(data.test)
```

---

## 🧪 Contributing

We welcome contributions! To get started:

1. Fork this repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ❤️ Acknowledgments

Special thanks to the developers and researchers who inspired this project. ChronosForge aims to provide tools for modern research and practical applications in the vast field of machine learning and probabilistic inference.
