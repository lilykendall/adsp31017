# Assignment 2: Lily Kendall

import matplotlib.pyplot as plt
import pandas
from datetime import datetime
import random
from sklearn import metrics
from matplotlib.ticker import MultipleLocator
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score, silhouette_score)
import numpy

# load dataset
df = pandas.read_excel("Grocery_Runs.xlsx")

### Question 1 (20 points)
# Let us explore how to perform an RFM analysis by dividing customers into quintiles. For each customer,
# you'll look at Recency, the maximum number of days since 2024-12-31, along with the median number
# of items purchased (Frequency), and the total transaction amounts (Monetary). Even if some
# transactions show zero or negative item counts or quantities, don't worry — you'll include all of them in
# your analysis to get a complete picture.

# (a) (10 points) What are the quintiles for Recency, Frequency, and Monetary values?

# create column for days since 2024-12-31
t_date = pandas.to_datetime(df['Transaction_Date'])
reference_date = datetime.strptime('2024-12-31', "%Y-%m-%d")
n_days = pandas.Series((t_date - reference_date) / numpy.timedelta64(1, 'D'), name = 'Recency')

# Create the training data
train_data = df[['Customer_ID', 'Transaction_Date', 'Transaction_Amount']].join(n_days)
train_data.head()

# Define the aggregation procedure outside of the groupby operation
aggregations = {
'Recency':'max',
'Customer_ID': 'count',
'Transaction_Amount': 'sum'
}

column_map = {'Recency': 'Recency', 'Customer_ID': 'Frequency', 'Transaction_Amount': 'Monetary'}
customer_data = train_data.groupby('Customer_ID').agg(aggregations).rename(columns =
column_map)
rfm_names = customer_data.columns

# Determine the quintiles
quintile = customer_data[rfm_names].describe(percentiles=[0.2, 0.4, 0.6, 0.8])
quintile

# (b) (10 points) Generate a separate vertical bar chart for each Recency, Frequency, and Monetary
# group. Display the three bar charts in the same chart frame.

# Assign the customer groups
customer_group = pandas.DataFrame(numpy.where(numpy.isnan(customer_data),0, 1), index = customer_data.index, columns = customer_data.columns)
for q in ['20%', '40%', '60%', '80%']:
    customer_group = customer_group + numpy.where(customer_data[rfm_names] > quintile.loc[q][rfm_names],1,0)
customer_group = customer_group.rename(columns = {'Recency': 'Recency Group', 'Frequency': 'Frequency Group', 'Monetary': 'Monetary Group'})
customer_group.head()

# Inspect bar charts of each group
groups = ['Recency Group', 'Frequency Group', 'Monetary Group']
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=200, sharey=True)

for i, g in enumerate(groups):
    group_size = customer_group[g].value_counts().sort_index()
    ax = axes[i]
    ax.bar(group_size.index, group_size, color='royalblue', zorder=3)
    ax.set_xlabel(g, fontweight='bold')
    if i == 0:
        ax.set_ylabel('Number of Customers')   
    ax.set_xticks(range(1, 6))
    
    # Grid and Locators
    ax.yaxis.set_major_locator(MultipleLocator(base=50))
    ax.yaxis.set_minor_locator(MultipleLocator(base=10))
    ax.yaxis.grid(True, linestyle='-', alpha=0.7, zorder=0)

plt.tight_layout()
plt.show()

### Question 2 (20 points)
# Once you've rescaled the Recency, Frequency, and Monetary values to a range of 0 to 100, you'll perform k-means clustering on these rescaled numbers. 

# rescale values and create training set for kmeans clustering
traindata = customer_data[['Recency', 'Frequency', 'Monetary']].copy()
# Formula: ((x - min) / (max - min)) * 100
traindata = (traindata - traindata.min()) / (traindata.max() - traindata.min()) * 100
traindata = traindata.reset_index(drop=True)


## This code is taken from Dr. Lam from in class example file "Week 4 KMeans 2D Example.py"
def getPositionSRS (nObs, nCluster):
    '''Get positions of centroids for the clusters by simple random sampling method
    Arguments:
    1. trainData - training data
    2. nObs - number of observations in training data (assume nObs > nCluster)
    3. nCluster - number of clusters
    Output:
    1. centroid_pos - positions in training data chosen as centroids
    '''
    centroid_pos = []
    kObs = 0
    iSample = 0
    for iObs in range(nObs):
        kObs = kObs + 1
        uThreshold = (nCluster - iSample) / (nObs - kObs + 1)
        if (random.random() < uThreshold):
            centroid_pos.append(iObs)
            iSample = iSample + 1
        if (iSample == nCluster):
            break
    return (centroid_pos)


def assignMember (trainData, centroid, distType):
    '''Assign observations to their nearest clusters.
    Arguments:
    1. trainData - training data
    2. centroid - centroid
    3. distType - distance metric
    Output:
    1. member - cluster memberships
    2. wc_distance - distances of observations to the nearest centroid
    '''
    pair_distance = metrics.pairwise_distances(trainData, centroid, metric =
    distType)
    member = pandas.Series(numpy.argmin(pair_distance, axis = 1), name = 'Cluster')
    wc_distance = pandas.Series(numpy.min(pair_distance, axis = 1), name =
    'Distance')
    return (member, wc_distance)

def KMeansCluster (trainData, nCluster, distType = 'euclidean', nIteration = 500, nTrial = 10, randomSeed = None):
    n_obs = trainData.shape[0]
    if (randomSeed is not None):
        random.seed(a = randomSeed)

    list_centroid = []
    list_wcss = []

    for iTrial in range(nTrial):
        centroid_pos = getPositionSRS(n_obs, nCluster)
        centroid = trainData.iloc[centroid_pos]
        member_prev = pandas.Series([-1] * n_obs, name = 'Cluster')

        for iter in range(nIteration):
            member, wc_distance = assignMember(trainData, centroid, distType)
            centroid = trainData.join(member).groupby(by = ['Cluster']).mean()
            member_diff = numpy.sum(numpy.abs(member - member_prev))
            if (member_diff > 0):
                member_prev = member
            else:
                break
        print(centroid)

        list_centroid.append(centroid)
        list_wcss.append(numpy.sum(numpy.power(wc_distance,2)))

    best_solution = numpy.argmin(list_wcss)
    centroid = list_centroid[best_solution]
    member, wc_distance = assignMember(trainData, centroid, distType)
    return (member, centroid, wc_distance)


# (a)	(10 points) Try the number of clusters between 2 and 15, inclusive, then plot the Elbow values, 
# the Silhouette Scores, the Calinski and Harabasz Scores, and the Davies-Bouldin Scores versus the number of clusters. 
# Your initial random seed is 710136264; add 31 after each trial.

# Initialize storage for metrics
nClusters = []
Elbow = []
Silhouette = []
CH_score = []
DB_score = []
TotalWCSS = []

max_nCluster = 15

for k_idx in range(max_nCluster):
    nCluster = k_idx + 1
    
    # Generate seed
    current_seed = 710136264 + (31 * k_idx)
    
    # Run KMeans function
    member, centroid, wc_distance = KMeansCluster(
        traindata, 
        nCluster, 
        distType='euclidean', 
        nTrial=20, 
        randomSeed=current_seed
    )
    
    # Calculate Total WCSS and Elbow (Mean WCSS)
    T = numpy.sum(numpy.power(wc_distance, 2))
    E = numpy.mean(numpy.power(wc_distance, 2))
    
    # Metrics requiring at least 2 clusters
    if nCluster > 1:
        S = metrics.silhouette_score(traindata, member, metric='euclidean')
        CH = metrics.calinski_harabasz_score(traindata, member)
        DB = metrics.davies_bouldin_score(traindata, member)
    else:
        # Defaults for k=1
        S, CH, DB = numpy.nan, numpy.nan, numpy.nan
        
    # Append results ONCE per k value
    nClusters.append(nCluster)
    TotalWCSS.append(T)
    Elbow.append(E)
    Silhouette.append(S)
    CH_score.append(CH)
    DB_score.append(DB)

# Create the Summary DataFrame
result_df = pandas.DataFrame({
    'N Cluster': nClusters,
    'Total WCSS': TotalWCSS,
    'Elbow': Elbow,
    'Silhouette': Silhouette,
    'CH Score': CH_score,
    'DB Score': DB_score
})

# I used Gemini to generate the code to plot the values

# Ensure we are only plotting for nCluster >= 2 for the evaluation scores
# (Silhouette, CH, and DB are not defined for k=1)
plot_df = result_df[result_df['N Cluster'] >= 2]

metrics_to_plot = [
    ('Total WCSS', 'Total Within-Cluster Sum of Squares (WCSS)'),
    ('Elbow', 'Elbow Value (Mean WCSS)'),
    ('Silhouette', 'Silhouette Score (Higher is Better)'),
    ('CH Score', 'Calinski-Harabasz Score (Higher is Better)'),
    ('DB Score', 'Davies-Bouldin Score (Lower is Better)')
]

for column, title in metrics_to_plot:
    plt.figure(figsize=(8, 5))
    
    # Use the full range for WCSS/Elbow, but 2-15 for the others
    data_to_use = result_df if 'WCSS' in column or 'Elbow' in column else plot_df
    
    plt.plot(data_to_use['N Cluster'], data_to_use[column], 
             marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(range(1, 16))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight the 2-15 range
    plt.axvspan(2, 15, color='gray', alpha=0.1, label='Target Range (2-15)')
    
    plt.tight_layout()
    plt.show()


### Question 3 (20 points)

# Use the t-Distributed Stochastic Neighbor Embedding (t-SNE) method to project the scaled Recency, Frequency, 
# and Monetary values into a two-dimensional space.  You will try perplexity values between 5 and 50 in increments of 5.  
# Your initial random seed is 710136202; add 31 after each trial.

# This is code taken from Dr. Lam in the file "Week 4 Visualize High Dimensional Data.py"

# Calculate the elbow value for a given cluster solution
def elbow_value (X, cluster_member, cluster_centroid):
    n_cluster = cluster_centroid.shape[0]
    WCSS = [0.0] * n_cluster
    cluster_count = [0.0] * n_cluster
    irow = 0
    for index, row in X.iterrows():
        k = cluster_member[irow]
        diff = row - cluster_centroid[k]
        WCSS[k] += diff.dot(diff)
        cluster_count[k] += 1.0
        irow += 1
        elbow = 0.0
    for k in range(n_cluster):
        if (cluster_count[k] > 0.0):
            elbow += WCSS[k] / cluster_count[k]
    return (elbow)

# Suggest the best guess for the number of clusters
def suggest_n_clusters(train_data, test_data=None, k_choices=range(2,11,1), random_seed=710136264):
    aseed = random_seed
    result_list = []
    # Define base metric names first
    metric_name = ['N Clusters', 'Elbow Train', 'Silhouette Train', 'CH Train', 'DB Train']

    for k in k_choices:
        aseed += 31
        # Using KMeans from sklearn
        obj_kmeans = KMeans(n_clusters=k, init='random', n_init='auto', random_state=aseed)
        cluster_train = obj_kmeans.fit(train_data)
        
        # Calculate training metrics
        elbow_train = obj_kmeans.inertia_ # Often used for Elbow value
        silhouette_train = silhouette_score(train_data, cluster_train.labels_)
        ch_train = calinski_harabasz_score(train_data, cluster_train.labels_)
        db_train = davies_bouldin_score(train_data, cluster_train.labels_)
        
        this_result = [k, elbow_train, silhouette_train, ch_train, db_train]
        
        if test_data is not None:
            test_labels = cluster_train.predict(test_data)
            elbow_test = elbow_value(test_data, test_labels, cluster_train.cluster_centers_)
            silhouette_test = silhouette_score(test_data, test_labels)
            ch_test = calinski_harabasz_score(test_data, test_labels)
            db_test = davies_bouldin_score(test_data, test_labels)
            # Append test metrics to the result list
            this_result.extend([elbow_test, silhouette_test, ch_test, db_test])
            
        result_list.append(this_result)

    # If test data was provided, update the column headers
    if test_data is not None:
        metric_name.extend(['Elbow Test', 'Silhouette Test', 'CH Test', 'DB Test'])

    metric_df = pandas.DataFrame(result_list, columns=metric_name)
    return metric_df

# Display graphs of metrics of the training data
def show_cluster_metric (metric_df):
    fig, axs = plt.subplots(2, 2, figsize = (12,8), dpi = 200, sharex = True)
    fig.subplots_adjust(hspace = 0.1, wspace = 0.2)

    ax = axs[0,0]
    ax.plot(metric_df['N Clusters'], metric_df['Elbow Train'], marker = 'o', color = 'royalblue')
    ax.set_xlabel('')
    ax.set_ylabel('Elbow Value')
    ax.set_xticks(metric_df['N Clusters'])
    ax.grid(axis = 'both', linestyle = ':', linewidth = 0.5)

    ax = axs[0,1]
    ax.plot(metric_df['N Clusters'], metric_df['Silhouette Train'], marker = 'o',
    color = 'crimson')
    ax.set_xlabel('')
    ax.set_ylabel('Silhouette Score')
    ax.set_xticks(metric_df['N Clusters'])
    ax.grid(axis = 'both', linestyle = ':', linewidth = 0.5)

    ax = axs[1,0]
    ax.plot(metric_df['N Clusters'], metric_df['CH Train'], marker = 'o', color =
    'seagreen')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Calinski and Harabasz Score')
    ax.set_xticks(metric_df['N Clusters'])
    ax.grid(axis = 'both', linestyle = ':', linewidth = 0.5)

    ax = axs[1,1]
    ax.plot(metric_df['N Clusters'], metric_df['DB Train'], marker = 'o', color = 'turquoise')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Davies-Bouldin Score')
    ax.set_xticks(metric_df['N Clusters'])
    ax.grid(axis = 'both', linestyle = ':', linewidth = 0.5)
    plt.show()
    return None

# Visualize the possible number of clusters by the t-SNE method
fig, axs = plt.subplots(2, 5, figsize = (20,8), dpi = 200)
fig.subplots_adjust(hspace = 0.3)
i = 0
j = 0
aseed = 710136202
tsne_seed = []
for k in range(5,51,5):
    tsne_seed.append([k, aseed])
    obj_tsne = TSNE(n_components = 2, perplexity = k, random_state = aseed)
    tsne_0 = obj_tsne.fit(traindata)
    embeddings = tsne_0.embedding_
    aseed += 31
    ax = axs[i,j]
    ax.set_title('Perplexity = ' + str(k))
    ax.scatter(embeddings[:,0], embeddings[:,1], c = 'royalblue', s = 20, marker ='o')
    ax.set_xlabel('Embeddings 0')
    ax.set_ylabel('Embeddings 1')
    ax.grid(axis = 'both', linestyle = ':', linewidth = 1)
    j += 1
    if (j > 4):
        j = 0
        i += 1
plt.show()

# Question 4 (20 points)
# After choosing the perplexity value that results in the smallest Kullback–Leibler divergence, 
# you'll find meaningful clusters within the t-SNE embeddings.

# (a)	(10 points) Try the number of clusters between 2 and 15 inclusively, 
# then plot the Elbow values, the Silhouette Scores, the Calinski and Harabasz Scores, 
# and the Davies-Bouldin Scores versus the number of clusters. Your initial random seed is 
# 710136264; add 31 after each trial.
metric_df = (suggest_n_clusters(traindata, k_choices=range(2,16)))
metric_df.head()

show_cluster_metric(metric_df)

# Question 5 (20 points)

# Since our goal in this experiment is to see if we can simplify the 125 segments from the RFM analysis
# into a few groups without losing their unique qualities, you'll cross-tabulate the cluster memberships
# from Question 4 with each Recency, Frequency, and Monetary group.

# (a) (10 points) Show the three cross-tabulation tables. We prefer to place the cluster memberships on
# the rows and the RFM variables on the columns. The cell contents are row percentages (i.e., the
# percentages within a row should sum to 100%).

best_k = 3 
aseed = 710136264 + (31 * (best_k - 2)) # Aligning seed with trial logic

best_perplexity = 50
obj_tsne = TSNE(n_components=2, perplexity=best_perplexity, random_state=710136202)
tsne_embeddings = obj_tsne.fit_transform(traindata)

# Run KMeans on embeddings
final_kmeans = KMeans(n_clusters=best_k, init='random', n_init='auto', random_state=aseed)
tsne_clusters = final_kmeans.fit_predict(tsne_embeddings)

# Add the cluster labels to customer_group dataframe for cross-tabulation
customer_group['TSNE_Cluster'] = tsne_clusters

rfm_groups = ['Recency Group', 'Frequency Group', 'Monetary Group']

# Create crosstab tables
for group in rfm_groups:
    print(f"\n--- Cross-tabulation: TSNE_Cluster vs {group} ---")
    
    # Create the crosstab
    ctab = pandas.crosstab(customer_group['TSNE_Cluster'], 
                           customer_group[group], 
                           normalize='index') * 100
    
    # Formatting for readability
    pandas.options.display.float_format = '{:.2f}%'.format
    display(ctab)