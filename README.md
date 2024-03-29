# Mixture-Of-Experts

![Mixture of Experts](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/00_switch_transformer.png)

This repository contains an implementation of the **Mixture of Experts** model with a standard transformer decoder in PyTorch. The model is implemented in two variants: sequential and sparse in Pytorch.

## Introduction

The Mixture of Experts model is a powerful approach for handling complex data. It combines multiple expert modules, each responsible for processing a specific part of the input, to produce accurate predictions. In this implementation, we use a standard transformer decoder to model the interactions between the experts.

There are two parameters introduced in this concept:
- `num_of_experts` (number of experts) which is the total number of experts available 
- `top_k` which is the top k experts (e.g. 2 out of 8) experts determined by fit (given by the gating mechanism) which will weigh their prediction in the final output 


## Sequential Mixture of Experts

The sequential Mixture of Experts model is implemented to handle data sequentially. It consists of multiple expert modules, each responsible for processing a specific part of the input sequence. The outputs of the expert modules are combined using a gating mechanism to produce the final prediction.

The input is handled one token at a time, passed through the selected expert(s) and added to the output.

## Sparse Mixture of Experts

The sparse Mixture of Experts model is designed to handle sparse data. It uses a sparse paralelizable approach, where the output tokens are a weighted sum of the `top_k` selected experts but handled much faster. 

To run the sparse Mixture of Experts model, follow the same steps as for the sequential model mentioned above.

[Find more about this in the orignal paper](https://arxiv.org/pdf/2208.02813.pdf)