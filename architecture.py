import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Definir las capas de la CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  #3 to 32, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  #32 to 64, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  #64 to 128, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Kernel=2 indica que su ventana de pooling es de 2x2 y stride=2 indica que se moverá 2 píxeles evitando solapamiento
        """
        #OPCIÓN FULLY CONNECTED LAYERS
        
        # Ajustar el tamaño de entrada para la capa totalmente conectada
        # Después de 3 MaxPooling, las dimensiones serán 32 / 2^3 = 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # (128 * 4 * 4)
        self.fc2 = nn.Linear(512, 2)  # 2 clases: REAL o FAKE
        
        #Esto no sé si queremos hacerlo en el decoder 
        """
        
        """
        OPCIÓN CLASIFICADOR CONVOLUCIONAL
        
        # Capa de Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Reduce a tamaño 1x1

        # Capa de salida
        self.fc_out = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)  # 2 clases: REAL y FAKE

        """
    def forward(self, x):
        # Aplicar las capas de convolución, activaciones y pooling 
        # Uso la función de activación de Relu
        x = self.pool(F.relu(self.conv1(x)))  # (batch_size, 32, 32) -> (batch_size, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (batch_size, 64, 16, 16) -> (batch_size, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # (batch_size, 128, 8, 8) -> (batch_size, 128, 4, 4)

        """
        #OPCIÓN FULLY CONNECTED LAYERS
        
        # Aplanar el tensor
        x = x.view(-1, 128 * 4 * 4)  # (batch_size, 128 * 4 * 4)
        
        # Capas totalmente conectadas
        x = F.relu(self.fc1(x))  # (batch_size, 512)
        x = self.fc2(x)  # (batch_size, 2)
        """
        
        """
        OPCIÓN CLASIFICADOR CONVOLUCIONAL
        
        # Aplicar Global Average Pooling
        x = self.global_avg_pool(x)  # (batch_size, 128, 4, 4) -> (batch_size, 128, 1, 1)

        # Capa de salida (Conv2D)
        x = self.fc_out(x)  # (batch_size, 2, 1, 1)

        # Aplanar el tensor
        x = x.view(-1, 2)  # (batch_size, 2)
        """
        return x
