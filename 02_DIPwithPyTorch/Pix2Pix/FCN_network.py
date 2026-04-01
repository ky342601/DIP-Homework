import torch.nn as nn
import torch

class FullyConvNetwork(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, num_filters=64):
        super().__init__()

        self.enc1 = self.conv_block(in_channels, num_filters, batch_norm=False, activation='leaky')  
        self.enc2 = self.conv_block(num_filters, num_filters * 2, batch_norm=True, activation='leaky') 
        self.enc3 = self.conv_block(num_filters * 2, num_filters * 4, batch_norm=True, activation='leaky') 
        self.enc4 = self.conv_block(num_filters * 4, num_filters * 8, batch_norm=True, activation='leaky') 
        self.enc5 = self.conv_block(num_filters * 8, num_filters * 8, batch_norm=True, activation='leaky') 
        self.enc6 = self.conv_block(num_filters * 8, num_filters * 8, batch_norm=True, activation='leaky') 
        self.enc7 = self.conv_block(num_filters * 8, num_filters * 8, batch_norm=True, activation='leaky') 
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_filters * 8, num_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ) 

        self.dec7 = self.deconv_block(num_filters * 8, num_filters * 8, batch_norm=True, dropout=True) 
        self.dec6 = self.deconv_block(num_filters * 16, num_filters * 8, batch_norm=True, dropout=True) 
        self.dec5 = self.deconv_block(num_filters * 16, num_filters * 8, batch_norm=True, dropout=True)
        self.dec4 = self.deconv_block(num_filters * 16, num_filters * 4, batch_norm=True, dropout=False) 
        self.dec3 = self.deconv_block(num_filters * 12, num_filters * 2, batch_norm=True, dropout=False) 
        self.dec2 = self.deconv_block(num_filters * 6, num_filters, batch_norm=True, dropout=False)     
        self.dec1 = self.deconv_block(num_filters * 3, num_filters, batch_norm=True, dropout=False)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(num_filters * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        '''self.encoder = nn.Sequential(
  
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        ### FILL: add more CONV Layers
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  
        )'''

    def conv_block(self, in_channels, out_channels, batch_norm=True, activation='relu'):

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
            
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky':
            layers.append(nn.LeakyReLU(0.2, inplace=True)) 
                
        return nn.Sequential(*layers)

    def deconv_block(self, in_channels, out_channels, batch_norm=True, dropout=False):

        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
                
        if dropout:
            layers.append(nn.Dropout(0.5)) 
                
        layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)


    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        
        bottleneck = self.bottleneck(e7)
        
        d7 = self.dec7(bottleneck)
        d7 = torch.cat([d7, e7], dim=1) 
        
        d6 = self.dec6(d7)
        d6 = torch.cat([d6, e6], dim=1) 
        
        d5 = self.dec5(d6)
        d5 = torch.cat([d5, e5], dim=1) 
        
        d4 = self.dec4(d5)
        d4 = torch.cat([d4, e4], dim=1) 
        
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1) 
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1) 
        
        d1 = self.dec1(d2) 
        d1 = torch.cat([d1, e1], dim=1) 
        
        output = self.final_layer(d1)
        
        return output
    