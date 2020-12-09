import torch 
import torch.nn as nn 
import torchvision.models as models 


class SimCLR(nn.Module):
    def __init__(self, embedding_dim):
        super(SimCLR, self).__init__()
        
        # output embedding length
        self.embedding_dim = embedding_dim
        
        # resnet50 base encoder
        self.f = nn.Sequential(
            *list(models.resnet50(pretrained=False).children())[:-1]
            )
        
        # mlp projection head
        self.g = ProjectionHead(embedding_dim=embedding_dim)
        
    def forward(self, x):
        h = self.f(x)
        h = torch.flatten(h, start_dim=1)
        z = self.g(h)
        return h, z
        
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim):
        super(ProjectionHead, self).__init__()
        self.embedding_dim = embedding_dim
        self.model = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim, bias=True)
        )
        
    def forward(self, x):
        z = self.model(x)
        return z


if __name__ == '__main__':  
    # batch size 3 with 64x64 image
    image_batch = torch.randn(3, 3, 64, 64)
    simclr = SimCLR(embedding_dim=128)
    h, z = simclr(image_batch)
    print(f"Representation shape: {h.size()}")
    print(f"Embedding shape: {z.size()}")