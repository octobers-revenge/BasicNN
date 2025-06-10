import torch
import torch.optim as optim
import torch.nn as nn
from NNmodel import NeuralNetwork 
from NNimage import trainloader, testloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork()
model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(5):  
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)        
        optimizer.zero_grad() 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

print("Finished Training")

model.eval()  # switch to evaluation mode
correct = 0
total = 0

with torch.no_grad():  
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")