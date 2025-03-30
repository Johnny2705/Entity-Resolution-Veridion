# Entity-Resolution-Veridion

## Overview

This project implements an entity resolution solution to identify and group duplicate company records imported from multiple systems. Due to variations in data many company entries may appear duplicated. The goal is to accurately detect these duplicates and distinguish them from unique company records.

## Methodology

My approach follows a systematic process:

1. **Data Preprocessing**
   Clean and standardize the data to ensure consistency across records.
2. **Composite Key Generation**
   Create a composite key for each record based on relevant attributes to facilitate comparison. I found the most relevant attributes to be:
   - Company Name
   - Legal Name
   - Commercial Names
   - Year Founded
   - Short Description
   - NAICS label
   - Generated business tags
     The composite key acts as a fingerprint for each record, allowing for efficient comparison. I tried to keep the composite key as short as possible while still trying to capture the most relevant information and minimizing the risk of false positives, this was taken into account for performance reasons. The bigger the composite key the more time it takes to compare records.
3. **Vectorization**
   To compare composite keys, I converted them into numerical features using TF-IDF vectorization, facilitating the use of nlp algorithms. I used character n-grams(ranging from 2 to 4 characters) to capture subtle differences in the text. This approach helps to identify similar records even when they have slight variations in spelling or formatting.
4. **Clustering with DBSCAN**
   I applied the DBSCAN algorithm to group similar records based on the cosine similarity of their TF-IDF vectors. I chose this algorithm because it handles noise effectively and it also scales well with high-dimensional data. The key parameters are eps(set to 0.2 in my code) and min_samples(set to 3 in my code). Eps is used to determine the maximum distance between points to be considered in the same cluster and min_samples is the minimum number of records required to form a dense cluster. I experimented with different values for these parameters, but the values mentioned above worked best for me and generated the most accurate results.

## Other Algorithms Tested

When trying to find an optimal solution I also tested the Agglomerative Clustering algorithm, but it proved to be less scalable for larger datasets due to its higher computational complexity. Agglomerative Clustering doesn't handle the noise, it tends to force every record into a cluster, even if they are not similar.Also, this algorithm requires the threshold to be set manually, rather than using a distance metric like DBSCAN.

Another algorithm I tried was KMeans, but again the number of clusters needs to be predetermined. KMeans could've been a good choice if the clusters were to be spherical and of similar size.

## Conclusion

The questions I asked myself were crucial in guiding my approach. Some of them like **Which attributes best represent a company's identity**, **How can I ensure accurate comparisons despite minor variations** or **What is the optimmal balance between detail and performance in the composite key** helped me improve my solution every time and avoid overfitting.
DBSCAN was my algorithm of choice because it provides a good balance between scalability and accuracy for the high-dimensional, sparse data text data.Its ability to handle noise and automatically determine the number of clusters proved essential for effective duplicate detetction.
Moving forward, I plan to further explore parameter tunning and feature engineering to enhance the performance of the solution.
