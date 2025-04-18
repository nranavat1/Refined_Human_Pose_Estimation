import torch
import torchvision
import torch.nn as nn


class FeaturePyramidNetwork(nn.Module):
    
    def __init__(self, in_channels=256, num_scales = 4, strides = [4,8,16,32]):
        super().__init__()
        
        assert num_scales == len(strides)
        
        self.in_channels = in_channels
        self.num_scales = num_scales
        
        #Bottom-up Pathway
        #making separate layers in case we need to keep track of the weights and biases
        self.bottom_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=strides[i])
            for i in range(num_scales)
        ])
        
        # Top-down pathway
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
            for _ in range(num_scales)
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            for _ in range(num_scales)
        ])
        
        
    def bottom_up_pathway(self, x):
        #building the ResNet feature
        #here with increasing spatial size
        
        residual_blocks = []
        input = x
        for i in range(self.num_scales):
            ci = self.bottom_convs[i](input)
            residual_blocks.append(ci)
            input = ci  
        
        return residual_blocks
    
    def top_down_pathway(self, residual_blocks):
        #go down and combine the residual blocks with the lateral connections
        
        merged_maps = []
        
        for i in range(self.num_scales-1, -1, -1):
            if i< self.num_scales-1:
                #Upsample the previous result to match the next result in the resnet
                upsampled = nn.functional.interpolate(
                                merged_maps[-1], mode='nearest', size = residual_blocks[i].shape[-2:]
                                )
                mi = self.lateral_convs[i](residual_blocks[i]) + upsampled
            else:
                mi = self.lateral_convs[i](residual_blocks[i]) #the last conv layer 
            merged_maps.insert(0, mi)
          
        feature_maps = []
        for map in merged_maps:
            pi = self.output_convs[i](map)
            feature_maps.append(pi)
        
        return feature_maps
            
    def forward(self, x):
        
        residuals = self.bottom_up_pathway(x)
        feature_maps = self.top_down_pathway(residuals)
        
        return feature_maps
        
        
        
        
        