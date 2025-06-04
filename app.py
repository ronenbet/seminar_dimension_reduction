import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits, load_wine, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import time
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Dimension Reduction Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📊 Dimension Reduction Visualizer")
st.markdown("""
This interactive web app allows you to explore and compare different dimension reduction algorithms:
- **PCA (Principal Component Analysis)**: Linear technique that finds directions of maximum variance
- **Random Projection**: Fast linear method using random matrices
- **t-SNE**: Non-linear technique great for visualization, preserves local structure
- **UMAP**: Non-linear technique that preserves both local and global structure
""")

# Sidebar for algorithm selection and parameters
st.sidebar.header("🛠️ Configuration")

# Dataset selection
st.sidebar.subheader("📄 Dataset Selection")
dataset_option = st.sidebar.selectbox(
    "Choose a dataset:",
    ["Iris", "Wine", "Digits", "Custom Blobs", "Upload CSV"]
)

# Load dataset based on selection
@st.cache_data
def load_dataset(dataset_name, n_samples=300, n_features=10, n_centers=4):
    if dataset_name == "Iris":
        data = load_iris()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
    elif dataset_name == "Wine":
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
        target_names = data.target_names
    elif dataset_name == "Digits":
        data = load_digits()
        X, y = data.data, data.target
        feature_names = [f"pixel_{i}" for i in range(X.shape[1])]
        target_names = [str(i) for i in range(10)]
    elif dataset_name == "Custom Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=n_centers, 
                         n_features=n_features, random_state=42, cluster_std=1.5)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        target_names = [f"cluster_{i}" for i in range(n_centers)]
    
    return X, y, feature_names, target_names

# Custom dataset parameters for blobs
if dataset_option == "Custom Blobs":
    st.sidebar.subheader("🎛️ Custom Dataset Parameters")
    n_samples = st.sidebar.slider("Number of samples", 100, 1000, 300)
    n_features = st.sidebar.slider("Number of features", 5, 50, 10)
    n_centers = st.sidebar.slider("Number of clusters", 2, 10, 4)
    X, y, feature_names, target_names = load_dataset(dataset_option, n_samples, n_features, n_centers)
elif dataset_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("Dataset shape:", df.shape)
        
        # Let user select target column
        target_col = st.sidebar.selectbox("Select target column (optional):", 
                                        ["None"] + list(df.columns))
        
        if target_col != "None":
            X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
            y = df[target_col].values
            feature_names = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()
        else:
            X = df.select_dtypes(include=[np.number]).values
            y = np.zeros(len(X))  # Dummy target
            feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
        
        target_names = [str(i) for i in np.unique(y)]
    else:
        st.warning("Please upload a CSV file or select a different dataset.")
        st.stop()
else:
    X, y, feature_names, target_names = load_dataset(dataset_option)

# Data preprocessing
st.sidebar.subheader("🔧 Preprocessing")
scale_data = st.sidebar.checkbox("Standardize features", value=True)

if scale_data:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X.copy()

# Algorithm selection
st.sidebar.subheader("🤖 Algorithm Selection")
algorithms = st.sidebar.multiselect(
    "Select algorithms to compare:",
    ["PCA", "Random Projection", "t-SNE", "UMAP"],
    default=["PCA", "t-SNE"]
)

if not algorithms:
    st.warning("Please select at least one algorithm from the sidebar.")
    st.stop()

# Algorithm parameters
algorithm_params = {}

for algo in algorithms:
    st.sidebar.subheader(f"⚙️ {algo} Parameters")
    
    if algo == "PCA":
        algorithm_params[algo] = {
            'n_components': st.sidebar.slider(f"{algo} - Components", 2, min(10, X.shape[1]), 2)
        }
    
    elif algo == "Random Projection":
        algorithm_params[algo] = {
            'n_components': st.sidebar.slider(f"{algo} - Components", 2, min(10, X.shape[1]), 2)
        }
    
    elif algo == "t-SNE":
        algorithm_params[algo] = {
            'n_components': st.sidebar.slider(f"{algo} - Components", 2, 3, 2),
            'perplexity': st.sidebar.slider(f"{algo} - Perplexity", 5, 50, 30),
            'learning_rate': st.sidebar.slider(f"{algo} - Learning Rate", 10, 1000, 200),
            'max_iter': st.sidebar.slider(f"{algo} - Max Iterations", 100, 2000, 1000)
        }
    
    elif algo == "UMAP":
        algorithm_params[algo] = {
            'n_components': st.sidebar.slider(f"{algo} - Components", 2, 3, 2),
            'n_neighbors': st.sidebar.slider(f"{algo} - Neighbors", 5, 100, 15),
            'min_dist': st.sidebar.slider(f"{algo} - Min Distance", 0.01, 1.0, 0.1),
            'metric': st.sidebar.selectbox(f"{algo} - Metric", 
                                         ['euclidean', 'manhattan', 'cosine'], index=0)
        }

# Apply dimension reduction algorithms
@st.cache_data
def apply_algorithm(algo_name, X_data, params):
    start_time = time.time()
    
    if algo_name == "PCA":
        model = PCA(**params)
        X_reduced = model.fit_transform(X_data)
        explained_variance_ratio = model.explained_variance_ratio_
        additional_info = {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': np.cumsum(explained_variance_ratio)
        }
    
    elif algo_name == "Random Projection":
        model = GaussianRandomProjection(**params, random_state=42)
        X_reduced = model.fit_transform(X_data)
        additional_info = {}
    
    elif algo_name == "t-SNE":
        model = TSNE(**params, random_state=42)
        X_reduced = model.fit_transform(X_data)
        additional_info = {}
    
    elif algo_name == "UMAP":
        model = umap.UMAP(**params, random_state=42)
        X_reduced = model.fit_transform(X_data)
        additional_info = {}
    
    computation_time = time.time() - start_time
    
    return X_reduced, computation_time, additional_info

# Display dataset information
st.header("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Samples", X.shape[0])
with col2:
    st.metric("Features", X.shape[1])
with col3:
    st.metric("Classes", len(np.unique(y)))
with col4:
    st.metric("Dataset", dataset_option)

# Show sample of the data
if st.checkbox("Show raw data sample"):
    df_display = pd.DataFrame(X[:10], columns=feature_names)
    df_display['Target'] = [target_names[i] for i in y[:10]]
    st.dataframe(df_display)

# Apply algorithms and create visualizations
st.header("🎨 Dimension Reduction Results")

results = {}
for algo in algorithms:
    with st.spinner(f"Applying {algo}..."):
        X_reduced, comp_time, additional_info = apply_algorithm(algo, X_scaled, algorithm_params[algo])
        results[algo] = {
            'X_reduced': X_reduced,
            'time': comp_time,
            'info': additional_info
        }

# Create visualizations
if len(algorithms) == 1:
    algo = algorithms[0]
    X_reduced = results[algo]['X_reduced']
    
    # Single algorithm visualization
    st.subheader(f"{algo} Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if X_reduced.shape[1] == 2:
            fig = px.scatter(
                x=X_reduced[:, 0], y=X_reduced[:, 1],
                color=[target_names[i] for i in y],
                title=f"{algo} - 2D Projection",
                labels={'x': 'Component 1', 'y': 'Component 2'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        else:  # 3D
            fig = px.scatter_3d(
                x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                color=[target_names[i] for i in y],
                title=f"{algo} - 3D Projection",
                labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'},
                color_discrete_sequence=px.colors.qualitative.Set1
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Computation Time", f"{results[algo]['time']:.3f}s")
        
        # Calculate silhouette score
        if len(np.unique(y)) > 1:
            sil_score = silhouette_score(X_reduced, y)
            st.metric("Silhouette Score", f"{sil_score:.3f}")
        
        # Additional algorithm-specific info
        if algo == "PCA" and results[algo]['info']:
            st.write("**Explained Variance Ratio:**")
            for i, ratio in enumerate(results[algo]['info']['explained_variance_ratio']):
                st.write(f"PC{i+1}: {ratio:.3f}")
            st.write(f"**Cumulative Variance:** {results[algo]['info']['cumulative_variance'][-1]:.3f}")

else:
    # Multiple algorithms comparison
    n_algos = len(algorithms)
    n_cols = min(2, n_algos)
    n_rows = (n_algos + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=algorithms,
        specs=[[{'type': 'scatter'} for _ in range(n_cols)] for _ in range(n_rows)]
    )
    
    colors = px.colors.qualitative.Set1[:len(np.unique(y))]
    
    for idx, algo in enumerate(algorithms):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        X_reduced = results[algo]['X_reduced']
        
        for class_idx, class_name in enumerate(target_names):
            mask = y == class_idx
            fig.add_trace(
                go.Scatter(
                    x=X_reduced[mask, 0],
                    y=X_reduced[mask, 1],
                    mode='markers',
                    name=f"{class_name}" if idx == 0 else None,
                    legendgroup=f"class_{class_idx}",
                    showlegend=idx == 0,
                    marker=dict(color=colors[class_idx % len(colors)]),
                    hovertemplate=f"<b>{class_name}</b><br>X: %{{x}}<br>Y: %{{y}}<extra></extra>"
                ),
                row=row, col=col
            )
    
    fig.update_layout(height=400 * n_rows, title_text="Algorithm Comparison")
    st.plotly_chart(fig, use_container_width=True)

# Performance comparison table
st.header("⏱️ Performance Comparison")
performance_data = []
for algo in algorithms:
    perf_data = {
        'Algorithm': algo,
        'Computation Time (s)': f"{results[algo]['time']:.3f}",
        'Components': algorithm_params[algo]['n_components']
    }
    
    # Add silhouette score if applicable
    if len(np.unique(y)) > 1:
        X_reduced = results[algo]['X_reduced']
        sil_score = silhouette_score(X_reduced, y)
        perf_data['Silhouette Score'] = f"{sil_score:.3f}"
    
    performance_data.append(perf_data)

performance_df = pd.DataFrame(performance_data)
st.dataframe(performance_df, use_container_width=True)

# Algorithm explanations
st.header("📚 Algorithm Explanations")
algo_explanations = {
    "PCA": """
    **Principal Component Analysis (PCA)**
    - **Type**: Linear dimensionality reduction
    - **Best for**: Understanding data variance, feature selection, preprocessing
    - **Pros**: Fast, interpretable, preserves global structure
    - **Cons**: Linear assumptions, may not capture non-linear relationships
    - **When to use**: When you need interpretable results or have linear relationships
    """,
    
    "Random Projection": """
    **Random Projection**
    - **Type**: Linear dimensionality reduction
    - **Best for**: Fast dimensionality reduction, preprocessing for other algorithms
    - **Pros**: Very fast, works well for high-dimensional data
    - **Cons**: Results are random, less interpretable
    - **When to use**: When speed is critical and you need quick dimensionality reduction
    """,
    
    "t-SNE": """
    **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
    - **Type**: Non-linear dimensionality reduction
    - **Best for**: Data visualization, exploring local structure
    - **Pros**: Excellent for visualization, preserves local neighborhoods
    - **Cons**: Slow, sensitive to parameters, can create false clusters
    - **When to use**: For visualization and exploring local patterns in data
    """,
    
    "UMAP": """
    **Uniform Manifold Approximation and Projection (UMAP)**
    - **Type**: Non-linear dimensionality reduction
    - **Best for**: Visualization, preserving both local and global structure
    - **Pros**: Faster than t-SNE, preserves more global structure, good for clustering
    - **Cons**: More complex parameter tuning, newer algorithm
    - **When to use**: When you need both local and global structure preservation
    """
}

for algo in algorithms:
    if algo in algo_explanations:
        st.markdown(algo_explanations[algo])
        st.markdown("---")

# Tips and recommendations
st.header("💡 Tips & Recommendations")
st.markdown("""
### Parameter Tuning Tips:

**PCA:**
- Use the explained variance ratio to choose the number of components
- Aim for 80-95% cumulative explained variance

**t-SNE:**
- Perplexity: Use 5-50, roughly related to the number of nearest neighbors
- Learning rate: 10-1000, higher for larger datasets
- Run multiple times with different random seeds

**UMAP:**
- n_neighbors: 2-100, smaller values preserve local structure
- min_dist: 0.0-1.0, smaller values create tighter clusters
- Try different distance metrics for different data types

### General Guidelines:
- Always standardize your data before applying algorithms
- PCA is great for exploratory data analysis and preprocessing
- Use t-SNE for final visualization after initial exploration
- UMAP is often the best balance between speed and quality
- Random Projection is excellent for preprocessing high-dimensional data
""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • [Source Code](https://github.com/)") 