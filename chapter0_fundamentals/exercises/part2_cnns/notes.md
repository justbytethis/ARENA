# 0.2 CNNs Study Note

## Making your own modules

> Q. Question - can you see what makes logits non-unique (i.e. why any given set of probabilities might correspond to several different possible sets of logits)?

Since softmax acts as a kind of normalization, adding the same constant to every logit does not change the resulting probabilities. In other words, for logits $\mathbf{x} = (x_1, \cdots, x_n)$ and shifted logits $\mathbf{x}' = (x_1 + a, \cdots, x_n + a)$,

$$
\frac{\exp (x_i + a)}{\Sigma_k \exp (x_k + a)} = 
\frac{\exp (x_i) \exp (a)}{\Sigma_k \exp (x_k) \exp(a)} =
\frac{\exp (x_i)}{\Sigma_k \exp (x_k)}.
$$

Thus, $\text{Softmax}(\mathbf{x}) = \text{Softmax}(\mathbf{x}')$.

## Training Neural Networks

> Question - can you explain why we include a data normalization function in torchvision.transforms?

If we do not normalize the data, converging while using gradient descent may be difficult.

> Question - what is the benefit of using shuffle=True when defining our dataloaders? What might the problem be if we didn't do this?

This enables the model to see data in random sequence, especially across epochs. If we do not turn on `shuffle` and original data is ordered in specific way, the model may get stuck in specific data distribution thus cannot learn from whole data in effective way.

## Convolutions

> Why would convolutional layers be less likely to overfit data than standard linear (fully connected) layers?

This is because of the parameter sharing of conv layers. Unlike vanila MLP layers, conv layers share kernles with weights and biases across all image region.

> Suppose you fixed some random permutation of the pixels in an image, and applied this to all images in your dataset, before training a convolutional neural network for classifying images. Do you expect this to be less effective, or equally effective?

It will be significantly less effective because CNNs utilize the inductive bias on locality. Thus if we permutate the pixels in an image and remove the locality, CNNs are no longer able to utilize it.

> If you have a 28x28 image, and you apply a 3x3 convolution with stride 2, padding 1, and 5 output channels, what shape will the output be?

(5, 14, 14).

## ResNets

> "Batch Normalization allows us to be less careful about initialization." Explain this statement.

We don't need to consider initialization methods such as Xavier or He initialization since we normalize the signals.

> Give at least 2 reasons why batch normalization improves the performance of neural networks.

1. The overall training process is more stable if we use BN, since model is more robust to extreme values.
2. Model also becomes more robust to the noise from (mini-)batch noise.

> If you have an input tensor of size (batch, channels, width, height), and you apply a batchnorm layer, how many learned parameters will there be?

To rescale for restoring representation power, we need `channels` parameters for scale parameters $\gamma$ and `channels` parameters for shift parameters $\beta$, thus `2 * channels` parameters in total.

> In the paper, the diagram shows additive skip connections (i.e. F(x) + x). One can also form concatenated skip connections, by "gluing together" F(x) and x into a single tensor. Give one advantage and one disadvantage of these, relative to additive connections.

- Advantage: Following components may organize the separate vectors from both F(x) and x thus preserve more information.
- Disadvantage: More parameters required for following components.

> Question: why do we not care about including biases in the convolutional layers?

Because of batch normalization followed by each conv layer. (No biases addition is meaningful.)