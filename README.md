# Lecture Notes
1. Unstructured Data: Audio Speech, Image, Text
2. **Only large NNs (=> large number of layers and weights) trained on large data show remarkably improved performance compared to small NNs trained on little data. Traditional learning algos like RFs, SVMs, linear regression may show better performance than small NNs trained on little data depending on feature engineering, parameter tuning etc. Training large NNs on large data became possible ONLY with large hardware like GPUs.** Prior to GPUs, it was not possible to train large NNs on large data and only small NNs could be trained on small data. These small NNs trained on small data didn't show any remarkable performance compared to traditional learning algos. So, traditionally, NNs were not considered attractive.
3.  Replacing the Sigmoid function with ReLU led to significant improvement in learning speed. Once the activation enters the plateau in the sigmoid function, further change in output slows down. But the ReLU has a decent gradient (=1).
4. **The R, G, B matrices of an image can be unrolled and concatenated in the form of long column matrix. This 1-D matrix can be input features to a NN.**
5. Dimensional Analysis:
      - **$\color{red}{\textrm{Dimension of the bias matrix (a vector) B[L], the activation (input) matrix and the output matrix of the L-th layer of an NN =}}$** $\color{red}{n(L)}$ **$\color{red}{\textrm{X}}$** $\color{red}{1}$ **$\color{red}{\textrm{, where}}$** $\color{red}{n(L)}$ **$\color{red}{\textrm{= number of neurons in L-th layer. So,}}$** $\color{red}{n(L)}$ **$\color{red}{\textrm{determines the number of rows.}}$**
      - **Dimension of the weight matrix W[L] (matrix containing all the weights of all the neurons in the L-th layer) of the L-th layer of an NN =** $\color{red}{n(L)}$ **$\color{red}{\textrm{X}}$** $\color{red}{n(L-1)}$ **. For the 1st layer,** $n(L-1)$ **= number of input features.**
      - **The above vectors - input activation and output of a NN layer - can be concatenated horizontally for all training samples to form columns of a matrix.**
6.  **Logistic regression is for performing binary classification.** 
    - **Logistic regression is done by a single neuron with sigmoid activation function ($\sigma(z)=1/1+e^{-z}=e^x/e^x+1$; $\sigma(z)$ $\epsilon$ (0,1)).**
      - **Sigmoid function is a probability distribution curve with its output limited to between 0 and 1. It is the probability of the output ($y$) to be 1 for a given $z$.** 
    - **Forward Propagation in logistic regression: $\hat{y}=\sigma(z)$ where, $z=w_1x_1+w_2x_2+b$**
    - **Non-convex optimization => multiple local minima. Squared error is not a suitable loss function for logistic regression because it is non-convex.**
    - **Preferred cost function ($J$) for logistic regression: Average of $-[(y_g)(log y)+(1-(y_g))log(1-y)]$ for all training samples. Gradient descent is used to find weights that reduce this cost function.  Here, $y_g$ = ground truth, $y$ = computed output.**
        - **$\color{red}{\textrm{Loss function is usually the negative natural log of the activation function.}}$**
        - **Here, the activation function is the combination ($\hat{y}^y(1-\hat{y})^{(1-y)}$) of the probability of y to be 1 ($=\hat{y}$) - and the probability of y to be 0 ($=1-\hat{y}$)**
        - **Derivation of the above cost function can be found [here](https://www.youtube.com/watch?v=k_S5fnKjO-4&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=24&ab_channel=DeepLearningAI).**
7. Gradient descent (GD) works by moving downward toward the pits or valleys in the graph ($J$ vs $w$ vs $b$; $J = f(y_g, y)$; $y_g$: ground truth; $y$: computed output) to find the minimum value.  **The gradient is a vector ($-dJ/dw$) that points in the direction of the steepest INCREASE of the function at a specific point. GD works by moving (step size=LR) in the opposite direction ($-dJ/dw$) of the gradient** allows the algorithm to gradually descend towards lower values of the function, and eventually reaching to the minimum of the function. BTW, "derivative" of a function at a point just means "slope" of the function at that point.
    - Back propagation (BP) implements gradient descent.
9. **GD for logistic regression: $\color{red}{w_i=w_i + (LR)(-dJ/dw_i)}$ where, $dJ/dw_i = (dJ/dy)(dy/dz)(dz/dw_i)$.**
    - **Value of $dJ/dw_i$ is different for different training samples - $z$, $y$.**
    - **Average all the values in the training set or in a mini-batch.**
10. $dy/dz$ for different activation functions(y):
    - sigmoid function: $z(1-z)$
    - tanh: $1-z^2$
    - ReLU: $0$ if $z<0$, $1$ if $z>0$
    - LeakyReLU: $1$ if $z>0$, some small value if $z<0$.
11. BP for a single neuron (with two weights $w_1$ and $w_2$) doing logistic regression:  $dJ/dw_1=(dJ/dy)(dy/dz)(dz/dw_1)$ where, $dJ/dy=-(y_g/y)+((1-y_g)/(1-y))$ and $dy/dz=y(1-y)$ and $dz/dw_1=x_1$. Here, $x_1$, $y_g$ are given and $y$ are known from forward propagation. Therefore, $w_1=w_1 + (LR)-dJ/dw1$ can be calculated. Similarly, $w_2$ (using, $dJ/dw_2=x_2(dJ/dz)$) and $b$ (using, $dJ/db=dJ/dz$) can be calculated. $dJ/dz=(dJ/dy)(dy/dz)$ can be calculated as above.
12. Gradient Descent is of 3 types: `Batch`, `Stochastic`, `Mini-batch`.
    - Batch gradient descent averages the errors for each sample in a training set, updating the model only after all training examples have been evaluated.
    - If the weights are updated for each training sample, its stochastic GD (SGD). **Because the coefficients are updated after every training instance, the updates will be noisy jumping all over the place, and so will the corresponding cost function. Its frequent updates can result in noisy gradients, but this can also be helpful in escaping the local minimum and finding the global one.**
        -  SGD often needs a small number of passes through the dataset to reach a good or good enough set of coefficients, e.g. 1-to-10 passes through the dataset.
    - In mini-batch GD, the training set is divided into subsets called mini-batches. The errors for all the samples in a mini-batch are averaged and used to compute the weight updates.The weights are updated with this average error at the end of a batch. This is done for each mini-batch in the training set.
        -  Mini-batch gradient descent combines concepts from both batch gradient descent and stochastic gradient descent. It splits the training dataset into small batch sizes and performs updates on each of those batches. This approach strikes a balance between the computational efficiency of batch gradient descent and the speed of stochastic gradient descent.
13. My thoughts:  Batch GD is faster compared to  SGD as the weights are updated only once per epoch. In case of multi-layered NNs, it means that backpropagation is done for each training sample. That's time consuming for each epoch.
14. Whenever possible, avoid using 'for' loops and nested 'for' loops in code. Use Vectors (1-D matrices) instead.
15. **NN is stacking multiple logistic regression one after another.**
    - No. of layers in NN=no. of layers of neurons
    - The inputs to the NN are considered as 0-th layer of activations
17.  Activation functions:
    - **$tanh(z)$** $= (e^z-e^{-z})/ (e^z+e^{-z})=$**$2\sigma(z) - 1$**
        -  **tanh is basically a stretched (along y axis) version of the sigmoid function -  stretched between +1 and -1**
        -  **$\color{red}{\textrm{tanh function always works better than sigmoid activation}}$**
        -  **sigmoid activation is always used in the output layer of a binary classification network because the output can only be 0/1. Never use it in any other case.**
    - **$\color{red}{\textrm{One disadvantage of both sigmoid and tanh activation functions is that when the  activations are too large or small, the gradient (dy/dz) get almost 0, thereby making the GD slow.}}$** Remember: $dJ/dw_i=(dJ/dy)(dy/dz)(dz/dw_i)$
    - One of the most popular activation functions: **ReLU = $max(0,z)$**

