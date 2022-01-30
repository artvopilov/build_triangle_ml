# BuildTriangleWithML
Building triangle using Logistic regression and simple Perceptron

### Data generation steps 
1. Define a triangle with 3 points
2. Generate random points in the triangle's area
3. Label points automatically

![triangle](https://github.com/mister025/BuildTriangleWithML/blob/main/triangle.png)

### Data preparations steps
1. Downsapmle negative examples
2. Split data into train and test

### Model 
1. Linear layer with 3 hidden neurons and sigmoid activation (aka 3 logistic legression)
2. Output linear layer with sigmoid activation
3. Log loss as loss function

### Training
1. Set learning rate and number of epochs
2. Use SGD optimizer 

![training](https://github.com/mister025/BuildTriangleWithML/blob/main/training.png)

### Evaluation (pictures attached)
1. Generate random points in the triangle's area
2. Label points with the model
3. Plot labelled points
 
![labelled points](https://github.com/mister025/BuildTriangleWithML/blob/main/model_labelled_data.png)
