import torch.nn as nn
import torch.nn.functional as F
import torch

# Identifier les dimensions manquantes dans l’architecture du réseau de neurones illustrée par la Figure 1.
# •Dans un nouveau fichier Python nn intro2.py, implémenter ce réseau de neurones convolutif avec PyTorch.
# •Instancier un nouveau réseau à partir de votre impl ́ementation et afficher son architecture.

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_fct = nn.Conv2d(in_channels=1, out_channels=10,kernel_size=5, padding=1, stride=1)
        self.conv2_fct = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=7, padding=0, stride=1)
        self.fc = nn.Linear(320, 10)
        
    def forward(self, x):
        x = self.conv1_fct(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        
        x = self.conv2_fct(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        
        return F.softmax(x, dim=1)
    
if __name__ == "__main__":
    model = Net()
    
    print("Model = ", model)
    # Vérification de taille
    x = torch.randn(1, 1, 30, 30)
    out = model(x)
    print("Taille de la sortie:", out.shape)
        
        
