
# Lecture Notes
1. Unstructured Data: Audio Speech, Image, Text
2. **A binary classifying neuron is called Perceptron.**
3. **Only large NNs (=> large number of layers and weights) trained on large data show remarkably improved performance compared to small NNs trained on little data. Traditional learning algos like RFs, SVMs, linear regression may show better performance than small NNs trained on little data depending on feature engineering, parameter tuning etc. Training large NNs on large data became possible ONLY with large hardware like GPUs.** Prior to GPUs, it was not possible to train large NNs on large data and only small NNs could be trained on small data. These small NNs trained on small data didn't show any remarkable performance compared to traditional learning algos. So, traditionally, NNs were not considered attractive.
4.  Replacing the Sigmoid function with ReLU led to significant improvement in learning speed. Once the activation enters the plateau in the sigmoid function, further change in output slows down. But the ReLU has a decent gradient (=1).
5. **The R, G, B matrices of an image can be unrolled and concatenated in the form of long column matrix. This 1-D matrix can be input features to a NN.**
6. Dimensional Analysis:
      - **$\color{red}{\textrm{Dimension of the bias matrix (a vector) B[L], the activation (input) matrix and the output matrix of the L-th layer
 of an NN =}}$** $\color{red}{n(L)}$ **$\color{red}{\textrm{X}}$** $\color{red}{1}$ **$\color{red}{\textrm{, where}}$** $\color{red}{n(L)}$ **$\color{red}{\textrm{= number of neurons in L-th layer. So,}}$** $\color{red}{n(L)}$ **$\color{red}{\textrm{determines the number of rows.}}$**
      - **Dimension of the weight matrix W[L] (matrix containing all the weights of all the neurons in the L-th layer) of the L-th layer of an NN =** $\color{red}{n(L)}$ **$\color{red}{\textrm{X}}$** $\color{red}{n(L-1)}$ **. For the 1st layer,** $n(L-1)$ **= number of input features.**
      - **The above vectors - input activation and output of a NN layer - can be concatenated horizontally for all training samples to form columns of a matrix.**
7.  **Logistic regression is for performing binary classification.** 
    - **Logistic regression is done by a single neuron with sigmoid activation function ($\color{red}{\sigma(z)=1/1+e^{-z}=e^x/e^x+1}$; $\sigma(z)$ $\epsilon$ (0,1)).**
      - **The logistic regression function is a probability distribution curve with its output limited to between 0 and 1. It is the probability of the output ($y$) to be 1 for a given $z$.**
      - **Any S-shaped function can be called sigmoid function. Logistic regression and $tanh$ are sigmoid functions.**  
    - **Forward Propagation in logistic regression: $\hat{y}=\sigma(z)$ where, $z=w_1x_1+w_2x_2+b$**
    - **Non-convex optimization => multiple local minima. Squared error is not a suitable loss function for logistic regression because it is non-convex.**
    - **Preferred cost function ($J$) for logistic regression: Average of $-[(y_g)(log y)+(1-(y_g))log(1-y)]$ for all training samples. Gradient descent is used to find weights that reduce this cost function.  Here, $y_g$ = ground truth, $y$ = computed output.**
        - **$\color{red}{\textrm{Loss function is usually the negative natural log of the activation function.}}$**
        - **Here, the activation function is the combination ($\hat{y}^y(1-\hat{y})^{(1-y)}$) of the probability of y to be 1 ($=\hat{y}$) - and the probability of y to be 0 ($=1-\hat{y}$)**
        - **Derivation of the above cost function can be found [here](https://www.youtube.com/watch?v=k_S5fnKjO-4&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=24&ab_channel=DeepLearningAI).**
8. Gradient descent (GD) works by moving downward toward the pits or valleys in the graph ($J$ vs $w$ vs $b$; $J = f(y_g, y)$; $y_g$: ground truth; $y$: computed output) to find the minimum value.  **The gradient is a vector ($-dJ/dw$) that points in the direction of the steepest INCREASE of the function at a specific point. GD works by moving (step size=LR) in the opposite direction ($-dJ/dw$) of the gradient** allowing the algorithm to gradually descend towards lower values of the function, and eventually reaching to the minimum of the function. BTW, "derivative" of a function at a point just means "slope" of the function at that point.
    - **Imagining this slope or gradient vector in a space with 10s of 1000s of dimensions (weights; a network with 784 neurons in 1st layer, 16 neurons in 2nd layer, 16 neurons in 3rd layer and 10 neurons in the final layer has a total of around 13k weights or dimensions) is beyond the perception of the human mind.**
    - Back propagation (BP) is an algorithm for computing gradient descent.
9. **GD for logistic regression: $\color{red}{w_i=w_i + (LR)(-dJ/dw_i)}$ where, $\color{red}{dJ/dw_i = (dJ/dy)(dy/dz)(dz/dw_i)}$.**
    - **$dy/dz$ = GRADIENT OF THE ACTIVATION FUNCTION**
    - **Value of $dJ/dw_i$ is different for different training samples - $z$, $y$.**
    - **Average all the values in the training set or in a mini-batch.**
10. BP for a single neuron (with two weights $w_1$ and $w_2$) doing logistic regression:  $\color{red}{dJ/dw_1=(dJ/dy)(dy/dz)(dz/dw_1)}$ where, $\color{red}{dJ/dy=-(y_g/y)+((1-y_g)/(1-y))}$ and $\color{red}{dy/dz=y(1-y)}$ and $\color{red}{dz/dw_1=x_1}$. Here, $x_1$, $y_g$ are given and $y$ are known from forward propagation. Therefore, $w_1=w_1 + (LR)-dJ/dw1$ can be calculated. Similarly, $w_2$ (using, $dJ/dw_{2}$ $=$ $x_{2}$ $(dJ/dz)$) and $b$ (using, $dJ/db=dJ/dz$) can be calculated. $dJ/dz=(dJ/dy)(dy/dz)$ can be calculated as above.
11. Gradient Descent is of 3 types: `Batch`, `Stochastic`, `Mini-batch`.
    - Batch gradient descent averages the errors for each sample in a training set, updating the model only after all training examples have been evaluated.
    - If the weights are updated for each training sample, its stochastic GD (SGD). **Because the coefficients are updated after every training instance, the updates will be noisy jumping all over the place, and so will the corresponding cost function. Its frequent updates can result in noisy gradients, but this can also be helpful in escaping the local minimum and finding the global one.**
        -  SGD often needs a small number of passes through the dataset to reach a good or good enough set of coefficients, e.g. 1-to-10 passes through the dataset.
    - In mini-batch GD, the training set is divided into subsets called mini-batches. The errors for all the samples in a mini-batch are averaged and used to compute the weight updates.The weights are updated with this average error at the end of a batch. This is done for each mini-batch in the training set.
        -  Mini-batch gradient descent combines concepts from both batch gradient descent and stochastic gradient descent. It splits the training dataset into small batch sizes and performs updates on each of those batches. This approach strikes a balance between the computational efficiency of batch gradient descent and the speed of stochastic gradient descent.
12. My thoughts:  Batch GD is faster compared to  SGD as the weights are updated only once per epoch. In case of multi-layered NNs, it means that backpropagation is done for each training sample. That's time consuming for each epoch.
13. Whenever possible, avoid using 'for' loops and nested 'for' loops in code. Use Vectors (1-D matrices) instead.
14. **NN is stacking multiple logistic regression one after another.**
    - No. of layers in NN=no. of layers of neurons
    - The inputs to the NN are considered as 0-th layer of activations
15. Activation functions:
    - **$\color{red}{tanh(z)}$ $\color{red}{= (e^z-e^{-z})/ (e^z+e^{-z})=2\sigma(2z) - 1}$**
        - **SO, $\sigma(x)$ CAN BE RE-SCALED (BY A FACTOR OF 2) AND SHIFTED (BY -1) TO OBTAIN $tanh(x)$. BOTH $tanh(x)$ AND $\sigma(x)$ ARE ESSENTIALLY THE SAME.**
          -  **Since $0 < \sigma(z) < 1$, $-1 < tanh(z) < 1$.**
        - **So, tanh is basically a stretched (along y axis) version of logistic regression -  stretched between +1 and -1**
        - **$\color{red}{\textrm{tanh FUNCTION ALWAYS WORKS BETTER THAN LOGISTIC REGRESSION.}}$**
          - **REASON: The outputs of the tanh function are closer to (or, centered around) 0 on average. The outputs of logistic regression are closer to (or, centered around) 0.5 on an average. "Convergence is usually faster if the average of each input variable over the training set is close to zero." - Yan LeCun. It has been long known (LeCun et al., 1998b; Wiesler & Ney, 2011) that the network training converges faster if its inputs are whitened – i.e., linearly transformed to have zero means and unit variances, and decorrelated. This is why you should normalize your inputs so that the average is zero. This heuristic should be applied at all layers which means that we want the average of the outputs of a node to be close to zero because these outputs are the inputs to the next layer. As each layer observes the inputs produced by the layers below, it would be advantageous to achieve the same whitening of the inputs of each layer.  So, if the activation units are $tanh$, then the hidden layers can converge faster.**
          - centered around zero is nothing but mean of the input data is around zero.
        -  **sigmoid activation is always used in the output layer of a binary classification network because the output can only be 0/1. Never use it in any other case.**
    - **$\color{red}{\textrm{One disadvantage of both sigmoid and tanh activation functions is that when the  activations are too large or small, the gradient (dy/dz) get almost 0, thereby making the GD slow.}}$** Remember: $dJ/dw_i=(dJ/dy)(dy/dz)(dz/dw_i)$
    - One of the most popular activation functions: **ReLU = $\color{red}{max(0,z)}$**
        - **ReLU preserves only the positive inputs and zeros the negative inputs.**
        - **ReLU is increasingly the default choice of activation function for hidden neurons in an NN.**
        - One disadvantage of ReLU is that it's derivative is 0 for negative $z$, leading to the **dying ReLU problem** and to almost stationary GD i.e., no update of weights and such a dead ReLU outputs only 0. **$\color{red}{\textrm{The gradient for -ve } z \textrm{ is 0 but in practice, enough of the hidden units have }z>0 \textrm{ and so, the learning can still progress well.}}$**
    - To overcome this, we have LeakyReLU ($\color{red}{=(0.01z, z)}$) which has a slight slope for negative $z$. 
        - **$\color{red}{\textrm{Leaky ReLU usually works better than ReLU}}$. However, either is fine. Leaky ReLU is not used much in practice.**
    - **Softmax is another activation function. This function maps the input numbers to output numbers between 0 and 1. This output number is a probability value. Higher the input number (activation), the higher the corresponding probability value. It is used in the output layer of an NN doing multi-class classification.** The output class with the highest probability is considered finally.
      - **$= e^{a_1}/\sum e^{a_i}$; here, $a_i$ is an input activation from the previous layer. All the activations from the previous layer must be connected to all the softmax units in the current layer, thereby, creating A FULLY CONNECTED LAYER.**
     
16. $dy/dz$ for different activation functions(y):
    - sigmoid function: $z(1-z)$
    - softmax function: $z(1-z)$
    - tanh: $1-z^2$
      - $\color{red}{tanh(z)}$ $\color{red}{= (e^z-e^{-z})/ (e^z+e^{-z})=2\sigma(2z) - 1 => tanh'(z) = 4\sigma'(z)}$ 
    - ReLU: $0$ if $z<0$, $1$ if $z>0$
    - LeakyReLU: $1$ if $z>0$, some small value if $z<0$
17. **Having linear activations or no non-linear activation function (i.e. the output is the activation - $wx+b$ - itself) for the hidden units of a multi-layered NN  =  having a single layer of linear or no activation, i.e., there will be no effect of having multiple neuron layers.** The combination of two or more linear functions is itself a linear function. Such an activation-function-less multi-layered NN - no matter how many layers it has - effectively behaves like a single-layered NN with identity or no activation function whose final output is a linear combination of the input. So, no effect of having multiple layers - in which each layer learns a new and different feature.
    - If we have a multi-layered NN with hidden layers having identity or no activation function and only the output layer having sigmoid activation function, such a NN is no more expressive than a standard logistic regression without any hidden layer.

18. If two or more neurons (in a layer) with same inputs and same output neurons have all their weights  initialised to zeros or same values,  then they will have the same weights even after N number of iterations of the BP.  Random initialization is done to avoid this symmetry between the weights of the neurons.

19. **Weights are initialized to small (random) values to avoid too large or small activations i.e. saturation (leading to slow gradient descent) of neurons with sigmoid or tanh activation functions.**
20. **The biases are usually initialized to zeros.**
21. There are 3 main ways of initializing the weights of an NN.
    - Glorot (Xavier) initialization: suitable for tanh, softmax, and logistic activation functions
    - He initialization: suitable for ReLU (and its variants) activation function
    - LeCun initialization: suitable for SELU activation.
    - Refer to this video for more information: https://www.youtube.com/watch?v=tYFO434Lpm0&list=PLcWfeUsAys2nPgh-gYRlexc6xvscdvHqX&index=9&ab_channel=AssemblyAI

22. **The weights and biases are the only parameters of an NN**. Rest all are hyperparameters. 
23. **Hyperparameters are the parameters that in turn affect or control the final values of the actual parameters - weights and biases**. Hyperparameters include: number of hidden layers in the network, number of units in a layer, LR,  number of epochs, mini-batch size, choice of activation units, regularization parameters, momentum parameters etc.
24. The initial layers of a deep NN detect simple features (like, edges of an image). The later layers combine some of these simple features to detect more complex features. For example, a face detecting NN's final layer detects face(s), its penultimate layer of neurons detects different parts of a face (e.g. eyes, nose, etc), the preceding layer of neurons detects simpler features/edges composing the eyes, nose etc. and so on. Takeaway is that a deep NN detects simple features and hierarchically combines them to detect/produce more complex features in the successive layers of neurons. Lower layers learn simpler features, and later layers learn complex features. This holds good for not just image classification but also for other domains. Example: In deep NN for speech recognition, the initial layers would learn some simple, low-level sound waveforms, the following layers would learn simple phonemes (like, the constituent sounds of a word) using the previous neuron layers, the following layers would learn the syllables of words, the following layers would learn words. Using these words, the NN can also learn to recognize phrases or sentences.
25. $J = f(y_g, y)$ where, $y=w_{1}x_{1}+w_{2}x_{2}+w_{3}x_{3}+.....w_{n}x_{n}$; $dJ/dw_i=(dJ/dy)(dy/dz)(dz/dw_i)$ where, $(dJ/dy)=-(y_g/y)+((1-y_g)/(1-y))$ and $(dy/dz)=y(1-y)$ and $(dz/dw_i)=x_i$. Here, $x_i$, $y_d$ are given and $y$ is known from forward propagation. So, the value of $dJ/dw_i$ can be calculated for each $w_i$. **THE MORE THE VALUE OF $dJ/dw_i$, THE MORE SENSITIVE THE LOSS FUNCTION IS TO THAT WEIGHT**. Example: if $dJ/dw_1$ = 3.2 and $dJ/dw_2$ = 0.1, the loss function would change 32x more for a change $\delta$ in $w_1$ than for the same change $\delta$ in $w_2$.
26. The **BEST-EVER EXPLANATION OF BP**: https://www.youtube.com/watch?v=kbGu60QBx2o&ab_channel=ritvikmath
    - **BP is all about chain rule. In BP, some of the components of the chain rule are stored or cached as they are calculated first. These cached components are the gradients $dz_{i+1}/dx_i$, $dy_{i+1}/dz_{i+1}$ of the later or deeper layers that are traversed first in BP.**
27. The **BEST-EVER EXPLANATION OF THE VANISHING GRADIENT PROBLEM**: https://www.youtube.com/watch?v=ncTHBi8a9uA&t=903s&ab_channel=ritvikmath
28. NumPy is a linear algebra library.
29. We rarely use the "math" library of Python for deep learning because the inputs of the functions are real numbers. In deep learning, we mostly use matrices and vectors. This is why Numpy is more useful.
30. Sigmoid function on a matrix or vector `X`: `s = 1 / (1 + np.exp(-x))
31. Gradient (`ds/dx`) of the sigmoid function `s` with respect to its input `X`: `ds = s * (1 - s)`
32. 

## Doables
1.  Read paper on Batch Normalization: https://arxiv.org/pdf/1502.03167.pdf
2.  Read paper on Efficient Backpropagation: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
3.  What is Batch Normalization?
4.  
