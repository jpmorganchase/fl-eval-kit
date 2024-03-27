# Federated Learning Evaluation Kit (FL-Eval-Kit)

The Federated Learning Evaluation Kit includes two components, the [Absolute Variation Distance](https://link.springer.com/chapter/10.1007/978-3-031-56066-8_20) and the [D-Rank metric](https://link.springer.com/chapter/10.1007/978-3-031-56066-8_21); both published in the Advances in Information Retrieval, 46th European Conference on Information Retrieval. 

## Absolute Variation Distance: An Inversion Attack Evaluation Metric for Federated Learning

This paper introduces the Absolute Variation Distance (AVD), a lightweight metric derived from total variation, to assess data recovery and information leakage in FL. Federated Learning (FL) has emerged as a pivotal approach for training models on decentralized data sources by sharing only model gradients. However, the shared gradients in FL are susceptible to inversion attacks which can expose sensitive information. While several defense and attack strategies have been proposed, their effectiveness is often evaluated using metrics that may not necessarily reflect the success rate of an attack or information retrieval, especially in the context of multidimensional data such as images. Traditional metrics like the Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR) and Mean Squared Error (MSE) are typically used as lightweight metrics, assume only pixel-wise comparison, but fail to consider the semantic context of the recovered data. AVD metric addresses these shortcomings.

## Ranking Distance Metric for Privacy Budget in Distributed Learning of Finite Embedding Data

In this study, we show how privacy in a distributed FL setup is sensitive to the underlying finite embeddings of the confidential data. We show that privacy can be quantified for a client batch that uses either noise, or a mixture of finite embeddings, by introducing a normalised rank distance (d-rank). This measure has the advantage of taking into account the size of a finite vocabulary embedding, and align the privacy budget to a partitioned space. We further explore the impact of noise and client batch size on the privacy budget and compare it to the standard $\epsilon$ derived from Local-DP.

## Instructions

The files `AVD_example.py` and `DP_IMDB_example.py` contain examples for the AVD and the D-rank metrics/papers. The user can execute the files to see the results of d-rank in IMDB data and of AVD in MNIST dataset.