import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
# Suppress joblib CPU warning
os.environ['LOKY_MAX_CPU_COUNT'] = '2'  # Set this to the number of CPU cores you want to use

##Data Preparation######
# Load dataset
df = pd.read_csv('Dataset.csv')

# Fill missing values (forward fill method)
#df.fillna(method='ffill', inplace=True)

# Encode categorical columns using LabelEncoder
le = LabelEncoder()

categorical_columns = ['gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'Churn']

for column in categorical_columns:
    df[column] = le.fit_transform(df[column])


# Save preprocessed dataset
df.to_csv('Data_Preparation/preprocessed_dataset.csv', index=False)


###Split Data into Training and Testing Sets####\
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

X = df.drop(['Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Save the splits
X_train.to_csv('Data_Preparation/X_train.csv', index=False)
X_test.to_csv('Data_Preparation/X_test.csv', index=False)
y_train.to_csv('Data_Preparation/y_train.csv', index=False)
y_test.to_csv('Data_Preparation/y_test.csv', index=False)



####Apply Scaling###########
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaled data
pd.DataFrame(X_train_scaled).to_csv('Data_Preparation/X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('Data_Preparation/X_test_scaled.csv', index=False)

###Clustering Analysis #######
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_train_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('Clustering_Analysis/elbow_method.png')

#####Train K-Means Model######
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X_train_scaled)

# Save the trained model using joblib
import joblib
joblib.dump(kmeans, 'Clustering_Analysis/kmeans_model.pkl')

########Cluster Visualization##########
import seaborn as sns

# Assign clusters to training data points
clusters_train = kmeans.predict(X_train_scaled)

# Create a new column 'Cluster' for training data in the original DataFrame
df_train = X_train.copy()  # Create a copy of the training data
df_train['Cluster'] = clusters_train

# Save the preprocessed dataset with clusters
df_train.to_csv('Data_Preparation/train_with_clusters.csv', index=False)

# Visualize clusters based on 'tenure' and 'MonthlyCharges'
sns.scatterplot(x=df_train['tenure'], y=df_train['MonthlyCharges'], hue=df_train['Cluster'], palette='coolwarm')
plt.title('Clusters based on Tenure and MonthlyCharges')
plt.savefig('Clustering_Analysis/clusters_visualization.png')
