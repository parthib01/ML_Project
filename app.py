import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Function to perform clustering and plot
def perform_clustering_and_plot(data, num_clusters):
    # Assuming the 'CustomerID' column is present and contains customer IDs
    # Assuming the 'StockCode' column represents product codes

    # ... (code to create customer_product_matrix and normalized_matrix as before)
    data = pd.read_csv('dataset.csv')
    data.dropna(inplace=True)
    customer_product_matrix = data.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='count', fill_value=0)
    normalized_matrix = customer_product_matrix.apply(lambda x: x / x.sum(), axis=1)


    # Calculate the distortion for a range of cluster numbers
    distortions = []
    for i in range(1, num_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(normalized_matrix)
        distortions.append(kmeans.inertia_)

    # Plot the elbow method graph
    elbow_fig = plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_clusters + 1), distortions, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    
    # Show the elbow method graph using st.pyplot()
    st.pyplot(elbow_fig)

    # Ask the user to select the number of clusters from the elbow method graph
    selected_num_clusters = st.slider("Select the number of clusters from the elbow method graph", min_value=1, max_value=num_clusters, value=3)

    # Apply k-means clustering with the selected number of clusters
    kmeans = KMeans(n_clusters=selected_num_clusters, random_state=0)
    clusters = kmeans.fit_predict(normalized_matrix)

    # Apply PCA for dimensionality reduction to visualize the clusters in 2D
    pca = PCA(n_components=2)
    reduced_matrix = pca.fit_transform(normalized_matrix)

    # Add the cluster labels to the reduced matrix
    reduced_df = pd.DataFrame(reduced_matrix, columns=['PC1', 'PC2'])
    reduced_df['Cluster'] = clusters

    # Get the coordinates of the cluster centroids
    cluster_centers = pca.transform(kmeans.cluster_centers_)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(reduced_df['PC1'], reduced_df['PC2'], c=reduced_df['Cluster'], cmap='viridis', alpha=0.7)
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', s=100, c='black', label='Cluster Centroids')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Clusters Visualization (PCA) with Cluster Centroids')
    ax.legend()
    st.pyplot(fig)

# Use Streamlit to create the web app
st.title('Cluster Analysis Web App')
st.write("Upload your own dataset (CSV format) with 'CustomerID' and 'StockCode' columns:")

# File upload widget
uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a pandas DataFrame
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded dataset:")
    st.write(data.head())  # Display the first few rows of the uploaded dataset
    
    # Allow users to choose the number of clusters using the elbow method
    num_clusters = st.slider("Choose the maximum number of clusters", min_value=2, max_value=20, value=10)
    
    # Perform clustering and plot with the chosen number of clusters
    perform_clustering_and_plot(data, num_clusters)
    
#In this updated code, the elbow method graph is created within a separate elbow_fig figure object, which is then passed to st.pyplot(elbow_fig) to display it using Streamlit's st.pyplot() function. This approach avoids the thread-safety issue associated with using the global figure object directly.





