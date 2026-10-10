"""
Utility functions for visualizing embeddings in 3D space.
"""

import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os


def visualize_embeddings_3d(embeddings, labels, title="3D Embedding Visualization", save_path=None):
    """
    Create 3D visualization of embeddings using PCA dimensionality reduction.
    
    Args:
        embeddings: Array of embedding vectors (n_samples, n_features)
        labels: List of labels for each embedding
        title: Title for the plot
        save_path: Optional path to save the figure. If None, saves as 'embedding_visualization.png'
    """
    # Reduce to 3D using PCA
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], 
               c='blue', marker='o', s=100)
    
    # Add labels
    for i, label in enumerate(labels):
        ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], 
                label, fontsize=10)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)
    
    # Save the figure (works in all environments)
    if save_path is None:
        save_path = f'embedding_visualization_{title}.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
   
    if matplotlib.get_backend() in ['TkAgg', 'Qt5Agg', 'Qt4Agg']:
        plt.show()
    else:
        plt.close(fig)  # Close to free memory in non-interactive environments

