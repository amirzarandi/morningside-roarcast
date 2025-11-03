# morningside-roarcast

We use Spatio-Temporal Graph Neural Networks (STGNNs) to model the S&P 100 stock market as a graph and predict stock price movements for portfolio selection.

This methodology is inspired by Pacreau, G., et al. (2021). *Graph Neural Networks for Asset Management*. [https://ssrn.com/abstract=3976168](https://ssrn.com/abstract=3976168)

## Setup & Run Instructions

1.  **Create a Virtual Environment:** use Python 3.12
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Data Notebooks (Order is important):**

    * **`data.ipynb`**: This notebook must be run first. It fetches the S&P 100 stock list from Wikipedia, collects 10 years of historical price data  and fundamental data from `yfinance`, and saves them to the `data/raw/` directory.
    * **`graph.ipynb`**: This notebook must be run second. It uses the raw data to construct the graph's adjacency matrix (`adj.npy`) based on shared sectors  and fundamentals correlation. This matrix is saved to `data/raw/`.

4.  **Run Model Notebooks:**
    Once the raw data and adjacency matrix are created, you can run the model notebooks. The `dataset/stock.py` file contains the `StocksDataset` class, which automatically processes the raw data into the required 3D PyTorch Geometric format and saves it to `data/processed/`.

    * **`trend.ipynb`**: Trains a **TGCN** model for a binary classification task (predicting 1-week up/down trend vs. the market).
    * **`forecasting.ipynb`**: Trains an **A3TGCN** model for a regression task (predicting the next day's normalized price) using an attention mechanism and MSE loss.
    * **`portfolio.ipynb`**: Loads the trained trend classifier and backtests its performance by creating Top-K portfolios.