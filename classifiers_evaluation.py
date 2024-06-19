from classifiers import Perceptron, GaussianNaiveBayes, LDA
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from math import atan2, pi
import numpy as np

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Callback function to capture loss values
        losses = []
        def callback(perceptron: Perceptron, Xi, yi):
            losses.append(perceptron._loss(X, y))

        # Fit perceptron
        Perceptron(callback=callback).fit(X, y)

        # Plot figure of loss as a function of fitting iteration
        iterations = list(range(1, len(losses) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=iterations, y=losses, mode='lines', name=n))

        # Update layout
        fig.update_layout(title='Perceptron Training Loss Progression Over Iterations on ' + n + ' Dataset',
                        xaxis_title='Iteration',
                        yaxis_title='Training Loss Value (Misclassification Error)',
                        legend_title='Dataset')

        # Save figure
        fig.write_image("perceptron_" + f.split(".")[0] + ".png")

        

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
            # Load dataset
            X, y = load_dataset(f)

            # Initialize classifiers
            gnb = GaussianNaiveBayes()
            lda = LDA()

            # Fit models and predict over training set
            gnb.fit(X, y)
            lda.fit(X, y)
            gnb_predictions = gnb.predict(X)
            lda_predictions = lda.predict(X)

            # Calculate accuracy
            from loss_functions import accuracy
            gnb_accuracy = accuracy(y, gnb_predictions)
            lda_accuracy = accuracy(y, lda_predictions)

            # Create subplots
            fig = make_subplots(rows=1, cols=2, subplot_titles=(
                f"Gaussian Naive Bayes\nAccuracy: {gnb_accuracy:.2f}",
                f"LDA\nAccuracy: {lda_accuracy:.2f}"
            ))
            fig.update_layout(title_text=f"Dataset: {f}")

            # Plot Gaussian Naive Bayes predictions
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                    marker=dict(color=gnb_predictions, symbol=y, showscale=True)),
                        row=1, col=1)

            # Plot LDA predictions
            fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                    marker=dict(color=lda_predictions, symbol=y, showscale=True)),
                        row=1, col=2)

            # Add Gaussian centers and ellipses for GNB
            for idx, mean in enumerate(gnb.mu_):
                fig.add_trace(go.Scatter(x=[mean[0]], y=[mean[1]], mode='markers', marker=dict(color='black', symbol='x')),
                            row=1, col=1)
                fig.add_trace(get_ellipse(mean, np.diag(gnb.vars_[idx])), row=1, col=1)

            # Add Gaussian centers and ellipses for LDA
            for idx, mean in enumerate(lda.mu_):
                fig.add_trace(go.Scatter(x=[mean[0]], y=[mean[1]], mode='markers', marker=dict(color='black', symbol='x')),
                            row=1, col=2)
                fig.add_trace(get_ellipse(mean, lda.cov_), row=1, col=2)

            fig.write_image("comparison_" + f.split(".")[0] + ".png")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()