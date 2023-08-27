# Lecture Notes
1. Unstructured Data: Audio Speech, Image, Text
2. Only large NNs (=> large number of layers and weights) trained on large data show remarkably improved performance compared to small NNs trained on little data. Traditional learning algos like RFs, SVMs, linear regression may show better performance than small NNs trained on little data depending on feature engineering, parameter tuning etc. Training large NNs on large data became possible ONLY with large hardware like GPUs. Prior to GPUs, it was not possible to train large NNs on large data and only small NNs could be trained on small data. These small NNs trained on small data didn't show any remarkable performance compared to traditional learning algos. So, traditionally, NNs were not considered attractive.
3.  Replacing the Sigmoid function with ReLU led to significant improvement in learning speed. Once the activation enters the plateau in the sigmoid function, further change in output slows down. But the ReLU has a decent gradient (=1).
4.  **Logistic regression is an algorithm for binary classification. Logistic regression is done by a single neuron with sigmoid activation function (sigma).** Forward Propagation in logistic regression: $z=w_1x_1+w_2x_2+b$; $y=sigma(z)$;
