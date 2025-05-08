import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import sys

df = pd.read_csv("data.csv")

#Create a DataFrame with all training data except the target column
X = df.drop("label", axis=1)
y = df["label"]

def ScaleData(X):
    #Create a StandardScaler object
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    #Convert the scaled data to a DataFrame
    x_scaled_df = pd.DataFrame(x_scaled, columns=X.columns)
    df_scaled = pd.concat([x_scaled_df, y], axis=1)

    return df_scaled

def knn_euclidean(data, k, i):
    #Create the KNN model with n neighbors and euclidean distance
    model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    model.fit(data)

    #Return distances and the indices of the nearest neighbors
    query_point = data.iloc[[i]]  # Use double brackets to keep DataFrame structure
    distances, indices = model.kneighbors(query_point)
    print("\nKNN Euclidean")
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
    return distances, indices

def knn_cosine(data, k, i):
    #Create the KNN model with n neighbors and cosine distance
    model = NearestNeighbors(n_neighbors=k, metric="cosine")
    model.fit(data)

    #Return distances and the indices of the nearest neighbors
    query_point = data.iloc[[i]]  # Use double brackets to keep DataFrame structure
    distances, indices = model.kneighbors(query_point)
    print("\nKNN Cosine")
    print(f"Distances: {distances}")
    print(f"Indices: {indices}")
    return distances, indices

#print(scaled_data.head())

def main(k, i):
    scaled_data = ScaleData(X)
    #print(scaled_data.head())
    print("\n=== Result with normalized data ===")
    # Use the same scaled data for both KNN calls
    knn_euclidean(scaled_data, k+1, i)
    knn_cosine(scaled_data, k+1, i)
    print("\n=== Result without normalized data ===")
    knn_euclidean(X, k+1, i)
    knn_cosine(X, k+1, i)

if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            print("Usage: python main.py <k_neighbors> <vector_index>")
            print("Example: python main.py 3 50")
            sys.exit(1)
            
        k = int(sys.argv[1])
        i = int(sys.argv[2])
        
        if k <= 0:
            raise ValueError("k must be positive")
        if i < 0 or i >= len(df):
            raise ValueError(f"vector_index must be between 0 and {len(df)-1}")
            
        main(k, i)
        
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("Both arguments must be integers and within valid ranges")
        print("Usage: python main.py <k_neighbors> <vector_index>")
        print("Example: python main.py 3 50")
        sys.exit(1)

#TODO E.M.D
