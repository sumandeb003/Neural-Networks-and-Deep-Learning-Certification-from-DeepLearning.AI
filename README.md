# Lecture Notes
1. Unstructured Data: Audio Speech, Image, Text
2. **Only large NNs (=> large number of layers and weights) trained on large data show remarkably improved performance compared to small NNs trained on little data. Traditional learning algos like RFs, SVMs, linear regression may show better performance than small NNs trained on little data depending on feature engineering, parameter tuning etc. Training large NNs on large data became possible ONLY with large hardware like GPUs.** Prior to GPUs, it was not possible to train large NNs on large data and only small NNs could be trained on small data. These small NNs trained on small data didn't show any remarkable performance compared to traditional learning algos. So, traditionally, NNs were not considered attractive.
3.  Replacing the Sigmoid function with ReLU led to significant improvement in learning speed. Once the activation enters the plateau in the sigmoid function, further change in output slows down. But the ReLU has a decent gradient (=1).
4.  **Logistic regression is for performing binary classification.** 
    - **Logistic regression is done by a single neuron with sigmoid activation function ($\sigma(z)=1/1+e^{-z}=e^x/e^x+1$; $\sigma(z)$ $\epsilon$ (0,1)).**
      - **Sigmoid function is a probability distribution curve with its output limited to between 0 and 1. It is the probability of the output ($y$) to be 1 for a given $z$.** 
    - **Forward Propagation in logistic regression: $\hat{y}=\sigma(z)$ where, $z=w_1x_1+w_2x_2+b$**
    - **Non-convex optimization => multiple local minima. Squared error is not a suitable loss function for logistic regression because it is non-convex.**
    - **Preferred cost function ($J$) for logistic regression: Average of $-[(y_g)(log y)+(1-(y_g))log(1-y)]$ for all training samples. Gradient descent is used to find weights that reduce this cost function.  Here, $y_g$ = ground truth, $y$ = computed output.**
        - **Loss function is usually the negative natural log of the activation function.**
        - **Here, the activation function is the combination of probability of y to be 1 ($=\hat{y}$) - and 0 ($=1-\hat{y}$)**
        - **Derivation of the above cost function can be found [here](https://www.youtube.com/watch?v=k_S5fnKjO-4&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0&index=24&ab_channel=DeepLearningAI).**
6.  **The R, G, B matrices of an image can be unrolled and concatenated in the form of long column matrix. This 1-D matrix can be input features to a NN.**
7.  Dimensional Analysis:
      - **Dimension of the bias matrix (a vector) B[L], the activation (input) matrix and the output matrix of the L-th layer of an NN =** $n(L)$ **X** $1$ **, where** $n(L)$ **= number of neurons in L-th layer. So,** $n(L)$ **determines the number of rows.**
      - **Dimension of the weight matrix W[L] (matrix containing all the weights of all the neurons in the L-th layer) of the L-th layer of an NN =** $n(L)$ **X** $n(L-1)$ **. For the 1st layer,** $n(L-1)$ **= number of input features.**
      - **The above vectors - input activation and output of a NN layer - can be concatenated horizontally for all training samples to form columns of a matrix.**
8. 
9. 
