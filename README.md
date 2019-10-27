# Cats-vs-Dogs-and-handling-overfitting

When there is small set of data the model sometimes learn from noises and unwanted data from the training examples , which causes 
negative performance for the model on new examples.this phenominan is called overfitting and model will have a hard tme in generalizing 
teh new data.


# Data augmentation
Overfitting generally occurs when there are a small number of training examples. One way to fix this problem is to augment the dataset so that it has a sufficient number of training examples. Data augmentation takes the approach of generating more training data from existing training samples by augmenting the samples using random transformations that yield believable-looking images. The goal is the model will never see the exact same picture twice during training. This helps expose the model to more aspects of the data and generalize better.

