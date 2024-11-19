import torch
import torch.nn as nn

class feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(feedforward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)
        
        
input_size = 10
hidden_size = [64, 32]
output_size = 1

model =  feedforward(input_size, hidden_size, output_size)
print(model)


criterion = nn.BCELoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X = torch.randn(100, input_size) 
print(X)
y = torch.randint(0, 2, (100, 1)).float()  

epochs = 100
for epoch in range(epochs):
    predictions = model(X)
    loss = criterion(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")