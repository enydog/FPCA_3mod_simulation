import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(123)

# Load data
print("Loading data...")
pd0 = pd.read_csv("GCdatcrop3.csv")
x_data = pd.read_csv("GCdat2time.csv")
x = x_data.iloc[:, 0].values

print(f"Original data shape: {pd0.shape}")
print(f"Time points: {len(x)}")

# Clean up the data set, crop at 2 hours, omit incomplete data sets
# Columns 4:93 in R correspond to columns 3:93 in Python (0-indexed)
p = pd0.iloc[:, 3:93].copy()
p = p.dropna()

print(f"Cleaned data shape: {p.shape}")

# Partition dataset (2000 training, rest testing)
train_size = min(2000, len(p))
train_indices = np.random.choice(len(p), size=train_size, replace=False)
test_indices = np.setdiff1d(np.arange(len(p)), train_indices)

train = p.iloc[train_indices].copy()
test = p.iloc[test_indices].copy()

print(f"Training set shape: {train.shape}")
print(f"Test set shape: {test.shape}")

# Get mean and SD for first and last columns
print(f"\nTraining data statistics:")
print(f"First column - Mean: {train.iloc[:, 0].mean():.3f}, SD: {train.iloc[:, 0].std():.3f}")
print(f"Last column - Mean: {train.iloc[:, -1].mean():.3f}, SD: {train.iloc[:, -1].std():.3f}")

# Log transform the data
ly = np.log(train.values)
lx = np.log(x)
lytest = np.log(test.values)

print(f"Log transformed training data shape: {ly.shape}")

class SimpleFPCA:
    """
    Simple Functional PCA implementation using standard PCA on discretized functions
    """
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.mu_ = None
        self.phi_ = None
        self.lambda_ = None
        self.work_grid_ = None
        
    def fit(self, X, t_grid=None):
        """
        Fit FPCA model
        X: array of shape (n_functions, n_time_points)
        t_grid: time grid points
        """
        self.work_grid_ = t_grid if t_grid is not None else np.arange(X.shape[1])
        
        # Center the data
        self.mu_ = np.mean(X, axis=0)
        X_centered = X - self.mu_
        
        # Apply PCA
        self.pca.fit(X_centered)
        
        # Store components (eigenfunctions) and eigenvalues
        self.phi_ = self.pca.components_.T  # Transpose to match R's phi format
        self.lambda_ = self.pca.explained_variance_
        
        return self
    
    def transform(self, X):
        """Transform data to PC scores"""
        X_centered = X - self.mu_
        return self.pca.transform(X_centered)
    
    def fit_transform(self, X, t_grid=None):
        """Fit and transform"""
        return self.fit(X, t_grid).transform(X)
    
    def predict(self, X, K=None):
        """Predict/reconstruct curves using K components"""
        if K is None:
            K = self.n_components
        
        # Get PC scores
        scores = self.transform(X)
        
        # Reconstruct using K components
        reconstruction = self.mu_ + scores[:, :K] @ self.phi_[:, :K].T
        
        return {
            'scores': scores,
            'pred_curves': reconstruction
        }

# Fit FPCA with 4 components
print("\nFitting FPCA...")
fpca_dense = SimpleFPCA(n_components=4)
fpca_dense.fit(ly, lx)

# Get PC scores for training data
train_scores = fpca_dense.transform(ly)

print(f"Explained variance ratios: {fpca_dense.pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(fpca_dense.pca.explained_variance_ratio_)}")

# Fit with K=3 components and calculate residuals
fit_result = fpca_dense.predict(ly, K=3)
fit_curves = fit_result['pred_curves']

# Undo log transformation
fit_curves_exp = np.exp(fit_curves)
train_exp = np.exp(ly)

# Calculate percentage residuals
residuals = 100 * (train_exp - fit_curves_exp) / fit_curves_exp

# Calculate mean and standard deviation of residuals across time
pres = np.mean(residuals, axis=0)
sdres = np.std(residuals, axis=0)

print(f"\nResiduals statistics:")
print(f"Mean absolute residual: {np.mean(np.abs(pres)):.3f}%")
print(f"Mean residual std: {np.mean(sdres):.3f}%")

# Visualization functions
def plot_raw_data_and_mean():
    """Plot raw MMPs and mean function"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot individual curves with transparency
    train_exp = np.exp(ly)
    for i in range(min(100, train_exp.shape[0])):  # Plot first 100 curves
        ax1.semilogx(x, train_exp[i], color='black', alpha=0.025)
    
    ax1.set_ylim(0, 2000)
    ax1.set_title('A. Individual Power Curves', loc='left')
    ax1.grid(True, alpha=0.3)
    
    # Plot mean function
    mean_curve = np.exp(fpca_dense.mu_)
    ax2.semilogx(x, mean_curve, color='black', linewidth=2)
    ax2.set_ylim(0, 2000)
    ax2.set_title('B. Mean Function', loc='left')
    ax2.grid(True, alpha=0.3)
    
    fig.supxlabel('Duration (s)')
    fig.supylabel('Power (W)')
    plt.tight_layout()
    return fig

def plot_principal_components():
    """Plot first 4 principal components"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    mean_curve = fpca_dense.mu_
    
    # Variance ranges for each component (similar to R code)
    ranges = [(-1.16, 1.16, 1160), (-0.5, 0.5, 500), (-0.2, 0.2, 200), (-0.14, 0.14, 140)]
    
    for comp_idx in range(4):
        ax = axes[comp_idx]
        
        # Plot mean function
        ax.semilogx(x, np.exp(mean_curve), color='black', linewidth=2)
        
        # Plot variations around mean
        n_min, n_max, n_steps = ranges[comp_idx]
        n_values = np.linspace(n_min, n_max, n_steps)
        
        # Create color gradient
        colors = plt.cm.Greys(np.linspace(0.3, 1.0, len(n_values)))
        
        for i, n in enumerate(n_values):
            variation = mean_curve + fpca_dense.phi_[:, comp_idx] * n
            ax.semilogx(x, np.exp(variation), color=colors[i], alpha=0.7, linewidth=0.5)
        
        ax.set_ylim(0, 1500)
        ax.set_title(f'FPC {comp_idx + 1}')
        ax.grid(True, alpha=0.3)
    
    fig.supxlabel('Duration (s)')
    fig.supylabel('Power (W)')
    plt.tight_layout()
    return fig

def plot_scree():
    """Create scree plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    explained_var = fpca_dense.pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    x_pos = np.arange(1, len(explained_var) + 1)
    
    ax.bar(x_pos, explained_var * 100, alpha=0.7, label='Individual')
    ax.plot(x_pos, cumulative_var * 100, 'ro-', label='Cumulative')
    
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Scree Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# Predict on test data
print("\nPredicting on test data...")
test_pred = fpca_dense.predict(lytest, K=3)
test_scores = test_pred['scores']
test_fit = test_pred['pred_curves']

# Undo log transformation
test_fit_exp = np.exp(test_fit)
test_exp = np.exp(lytest)

# Calculate test residuals
test_residuals = 100 * (test_exp - test_fit_exp) / test_fit_exp

# PC scores statistics for test data
print(f"\nTest PC scores statistics:")
for i in range(3):
    pc_scores = test_scores[:, i]
    print(f"PC{i+1} - Mean: {np.mean(pc_scores):.4f}, SD: {np.std(pc_scores):.4f}")

# Correlation matrix of PC scores
pc_corr = np.corrcoef(test_scores[:, :3].T)
print(f"\nPC scores correlation matrix:")
print(pc_corr)

# Simple runs test implementation
def runs_test(residuals):
    """Simple runs test for residuals"""
    # Convert to binary (above/below median)
    median_val = np.median(residuals)
    binary_seq = (residuals > median_val).astype(int)
    
    # Count runs
    runs = 1
    for i in range(1, len(binary_seq)):
        if binary_seq[i] != binary_seq[i-1]:
            runs += 1
    
    return runs

# Calculate runs test for test residuals
print("\nCalculating runs test...")
runs_stats = []
for i in range(test_residuals.shape[0]):
    runs = runs_test(test_residuals[i])
    runs_stats.append(runs)

runs_stats = np.array(runs_stats)
print(f"Runs test - Mean: {np.mean(runs_stats):.2f}, SD: {np.std(runs_stats):.2f}")

def plot_residuals():
    """Plot residuals analysis"""
    pres_test = np.mean(test_residuals, axis=0)
    sdres_test = np.std(test_residuals, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogx(x, pres_test, 'o', color='black', markersize=4)
    ax.errorbar(x, pres_test, yerr=2*sdres_test, fmt='none', color='black', capsize=2)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Percentage residuals')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_random_fits():
    """Plot 4 random fits"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Sample 4 random curves from test data
    random_indices = np.random.choice(test_exp.shape[0], 4, replace=False)
    
    for i, idx in enumerate(random_indices):
        ax = axes[i]
        
        # Plot observed data
        ax.semilogx(x, test_exp[idx], color='darkgray', linewidth=2, label='Observed')
        
        # Plot fitted curve
        ax.semilogx(x, test_fit_exp[idx], color='black', linewidth=2, label='Fitted')
        
        ax.set_ylim(0, 1750)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    fig.supxlabel('Duration (s)')
    fig.supylabel('Power (W)')
    plt.tight_layout()
    return fig

def plot_pc_distributions():
    """Plot PC score distributions"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot density estimates for first 3 PCs
    for i in range(3):
        pc_scores = train_scores[:, i]
        
        # Simple density estimation using histogram
        counts, bins = np.histogram(pc_scores, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        colors = ['black', 'blue', 'red']
        labels = [f'PC{i+1} (SD={np.std(pc_scores):.2f})' for i in range(3)]
        
        ax.plot(bin_centers, counts, color=colors[i], linewidth=2, label=labels[i])
    
    ax.set_xlabel('PC score')
    ax.set_ylabel('Density')
    ax.set_title('Kernel Density of PC Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 2)
    
    plt.tight_layout()
    return fig

# Generate all plots
print("\nGenerating plots...")

# 1. Raw data and mean function
fig1 = plot_raw_data_and_mean()
plt.show()

# 2. Principal components
fig2 = plot_principal_components()
plt.show()

# 3. Scree plot
fig3 = plot_scree()
plt.show()

# 4. Residuals plot
fig4 = plot_residuals()
plt.show()

# 5. Random fits
fig5 = plot_random_fits()
plt.show()

# 6. PC distributions
fig6 = plot_pc_distributions()
plt.show()

print("\nFPCA Analysis Complete!")
print(f"Training data: {ly.shape[0]} curves, {ly.shape[1]} time points")
print(f"Test data: {lytest.shape[0]} curves")
print(f"First 4 PCs explain {fpca_dense.pca.explained_variance_ratio_[:4].sum()*100:.1f}% of variance")
