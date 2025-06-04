# 📊 Dimension Reduction Visualizer

An interactive Streamlit web application for exploring and comparing different dimension reduction algorithms including PCA, Random Projection, t-SNE, and UMAP.

![App Screenshot](https://via.placeholder.com/800x400?text=Dimension+Reduction+Visualizer)

## 🚀 Features

### 🤖 Supported Algorithms
- **PCA (Principal Component Analysis)**: Linear technique that finds directions of maximum variance
- **Random Projection**: Fast linear method using random matrices  
- **t-SNE**: Non-linear technique great for visualization, preserves local structure
- **UMAP**: Non-linear technique that preserves both local and global structure

### 📊 Datasets Options
- **Built-in Datasets**: Iris, Wine, Digits
- **Custom Synthetic Data**: Generate blob clusters with configurable parameters
- **CSV Upload**: Upload your own datasets

### 🎨 Visualization Features
- Interactive 2D and 3D scatter plots
- Side-by-side algorithm comparison
- Real-time parameter tuning
- Performance metrics (computation time, silhouette score)
- Algorithm-specific insights (e.g., explained variance for PCA)

### ⚙️ Customization Options
- Feature standardization toggle
- Algorithm parameter adjustment
- Multiple algorithm comparison
- Export capabilities

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Poetry (recommended) or pip package manager

### 🎯 Option 1: Poetry Setup (Recommended)

[Poetry](https://python-poetry.org/) provides better dependency management and virtual environment handling.

1. **Install Poetry** (if not already installed)
   ```bash
   # On macOS/Linux
   curl -sSL https://install.python-poetry.org | python3 -
   
   # On Windows (PowerShell)
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   
   # Alternative: using pip
   pip install poetry
   ```

2. **Clone or download the project files**
   ```bash
   # If you have git
   git clone <repository-url>
   cd dimension-reduction-visualizer
   
   # Or simply create a new directory and copy the files
   mkdir dimension-reduction-visualizer
   cd dimension-reduction-visualizer
   ```

3. **Install dependencies and create virtual environment**
   ```bash
   poetry install
   ```

4. **Run the application**
   ```bash
   # Option 1: Using poetry run
   poetry run streamlit run app.py
   
   # Option 2: Activate the virtual environment first
   poetry shell
   streamlit run app.py
   
   # Option 3: Using the custom script
   poetry run run-app
   ```

5. **Open your browser**
   The app will automatically open at `http://localhost:8501`

### 🛠️ Option 2: Traditional pip Setup

1. **Clone or download the project files**
   ```bash
   # If you have git
   git clone <repository-url>
   cd dimension-reduction-visualizer
   
   # Or simply create a new directory and copy the files
   mkdir dimension-reduction-visualizer
   cd dimension-reduction-visualizer
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Activate on macOS/Linux
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   The app will automatically open at `http://localhost:8501`

## 🎯 Usage Guide

### Getting Started

1. **Select a Dataset**: Choose from built-in datasets or upload your own CSV
2. **Choose Algorithms**: Select one or more algorithms to compare
3. **Adjust Parameters**: Fine-tune algorithm parameters using the sidebar controls
4. **Visualize Results**: Explore the interactive plots and performance metrics

### Dataset Selection

#### Built-in Datasets
- **Iris**: Classic 4-feature flower classification dataset (150 samples)
- **Wine**: Wine chemistry dataset with 13 features (178 samples)  
- **Digits**: Handwritten digit recognition with 64 pixel features (1797 samples)
- **Custom Blobs**: Generate synthetic clustered data with customizable parameters

#### CSV Upload
- Upload any CSV file with numerical features
- Optionally specify a target column for classification visualization
- The app will automatically detect numerical columns

### Algorithm Parameters

#### PCA Parameters
- **Components**: Number of principal components to extract (2-10)

#### Random Projection Parameters  
- **Components**: Number of dimensions in the projection (2-10)

#### t-SNE Parameters
- **Components**: Output dimensions (2D or 3D)
- **Perplexity**: Balance between local and global aspects (5-50)
- **Learning Rate**: Step size for gradient descent (10-1000)
- **Max Iterations**: Maximum optimization iterations (100-2000)

#### UMAP Parameters
- **Components**: Output dimensions (2D or 3D)
- **Neighbors**: Size of local neighborhood (5-100)
- **Min Distance**: Minimum distance between points in embedding (0.01-1.0)
- **Metric**: Distance metric (euclidean, manhattan, cosine)

## 📈 Understanding the Results

### Visualizations
- **2D/3D Scatter Plots**: Points colored by class labels
- **Interactive Features**: Zoom, pan, hover for data point details
- **Side-by-side Comparison**: When multiple algorithms are selected

### Performance Metrics
- **Computation Time**: How long each algorithm took to run
- **Silhouette Score**: Measure of clustering quality (-1 to 1, higher is better)
- **Explained Variance** (PCA only): Proportion of variance captured by each component

### Algorithm-Specific Insights
- **PCA**: Shows explained variance ratio and cumulative variance
- **t-SNE**: Displays final KL divergence if available
- **UMAP**: Shows nearest neighbor graph statistics
- **Random Projection**: Provides projection matrix properties

## 🎛️ Advanced Features

### Data Preprocessing
- **Feature Standardization**: Normalize features to zero mean and unit variance
- **Automatic Data Type Detection**: Handles mixed data types in CSV uploads

### Comparison Tools
- **Multi-algorithm View**: Compare up to 4 algorithms simultaneously
- **Performance Table**: Tabular comparison of metrics across algorithms
- **Parameter Sensitivity**: Real-time updates when parameters change

### Educational Content
- **Algorithm Explanations**: Detailed descriptions of each method
- **Use Case Guidelines**: When to use each algorithm
- **Parameter Tuning Tips**: Best practices for parameter selection

## 🧪 Development

### Poetry Commands

```bash
# Install dependencies
poetry install

# Add a new dependency
poetry add package-name

# Add development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree

# Activate virtual environment
poetry shell

# Run commands in the environment
poetry run python script.py

# Build the package
poetry build

# Publish (if configured)
poetry publish
```

### Development Dependencies

The project includes several development tools configured in `pyproject.toml`:

- **pytest**: Testing framework
- **black**: Code formatter
- **flake8**: Linting
- **mypy**: Type checking
- **jupyter**: Notebook support

### Code Quality

```bash
# Format code
poetry run black .

# Lint code
poetry run flake8 .

# Type checking
poetry run mypy .

# Run tests
poetry run pytest
```

## 🔧 Troubleshooting

### Common Issues

1. **Poetry Installation Issues**
   ```bash
   # If poetry command not found, add to PATH or use:
   python -m poetry --version
   
   # Reinstall poetry
   curl -sSL https://install.python-poetry.org | python3 - --uninstall
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Virtual Environment Issues**
   ```bash
   # Reset poetry environment
   poetry env remove python
   poetry install
   
   # List environments
   poetry env list
   ```

3. **Import Errors**
   ```bash
   # With Poetry
   poetry install --sync
   
   # With pip
   pip install --upgrade -r requirements.txt
   ```

4. **Memory Issues with Large Datasets**
   - Try reducing the number of samples
   - Use Random Projection as preprocessing
   - Increase system memory or use cloud computing

5. **Slow Performance**
   - Reduce t-SNE iterations for faster results
   - Use fewer data points for initial exploration
   - Consider using UMAP instead of t-SNE for large datasets

6. **Parameter Sensitivity**
   - Start with default parameters
   - Make small incremental changes
   - Use multiple random seeds for stochastic algorithms

### Performance Tips
- **For Large Datasets**: Use Random Projection → UMAP pipeline
- **For Exploration**: Start with PCA to understand data structure
- **For Visualization**: Use t-SNE or UMAP with optimized parameters
- **For Speed**: Random Projection is fastest, PCA is fast and interpretable

## 📚 Technical Details

### Dependencies
- **streamlit**: Web app framework
- **plotly**: Interactive plotting
- **scikit-learn**: PCA, Random Projection, t-SNE, preprocessing
- **umap-learn**: UMAP algorithm implementation
- **pandas/numpy**: Data manipulation
- **seaborn/matplotlib**: Additional plotting utilities

### Architecture
- **Modular Design**: Separate functions for each algorithm
- **Caching**: Streamlit caching for expensive computations
- **Responsive UI**: Adapts to different screen sizes
- **Error Handling**: Graceful handling of edge cases

## 🤝 Contributing

Contributions are welcome! Here are some areas for improvement:

- [ ] Add more dimension reduction algorithms (Isomap, LLE, etc.)
- [ ] Implement batch processing for multiple datasets
- [ ] Add export functionality for plots and results
- [ ] Include more evaluation metrics
- [ ] Add animation for parameter changes
- [ ] Implement custom color schemes

### Development Setup

1. **Fork and clone the repository**
2. **Install Poetry** (see installation instructions above)
3. **Install dependencies**: `poetry install`
4. **Make your changes**
5. **Run tests**: `poetry run pytest`
6. **Format code**: `poetry run black .`
7. **Submit a pull request**

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Streamlit** team for the amazing web app framework
- **scikit-learn** contributors for robust ML implementations  
- **UMAP** developers for the excellent dimension reduction algorithm
- **Plotly** team for interactive visualization tools
- **Poetry** for excellent dependency management

---

**Built with ❤️ using Python, Streamlit, Poetry, and modern data science tools** 