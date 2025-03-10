![{8C6904D8-7C18-4A07-BA76-B9565B8F4033}](https://github.com/user-attachments/assets/a3d680c2-9fdd-4ad5-81cc-3ee7c8b881e3)# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model
![{3C843D8B-C50E-4A8D-97A6-96B603FC9D33}](https://github.com/user-attachments/assets/5219d873-8e39-4728-b79a-d2b3932ee882)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:SUSITHRA.B
### Register Number:212223220113
```
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history={'loss': []}
  def forward(self,x):
    x=self.relu(self.fc1(x)) 
    x=self.relu(self.fc2(x))
    x=self.fc3(x)  
    return x


# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        # Append loss inside the loop
        ai_brain.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```
## Dataset Information
![{E5F270B1-EA59-495B-869F-00154545D8BB}](https://github.com/user-attachments/assets/ecd947d1-4fcf-46f7-a7ed-1f65bece1cf3)
## OUTPUT
### Training Loss Vs Iteration Plot
![{8C6904D8-7C18-4A07-BA76-B9565B8F4033}](https://github.com/user-attachments/assets/add3248a-1ef8-4ac2-950f-027631a2f5a5)
### New Sample Data Prediction
![{FDD3F598-9ADB-43F9-91CA-4297B2585CE5}](https://github.com/user-attachments/assets/2194df11-5f7b-4339-913b-3f668842b7c4)

## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
