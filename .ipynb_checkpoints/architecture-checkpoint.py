import torch
import torch.nn as nn
import torch.nn.functional as F
        

class CNN_3C(nn.Module):
   
    def __init__(self,out_1, out_2, out_3, image_size):
        super(CNN_3C, self).__init__()
        
        self.out_1 = out_1 #Salida 1, normalización y entrada 2
        self.out_2 = out_2 # Salida 2, normalización y entrada 3
        self.out_3 = out_3 # Salida 3 y normalización
        self.image_size = int(image_size/ (2 ** 3)) #Tamaño nuevo después de 3 poolings
        
        '''
        Al crear la arquitectura definimos tres capas convolucionales 2d, siendo en la primera el canal de entrada 3 por los tres canales(RGB) que posee una foto, y 32 canales de salida porque así lo estipulamos.
        El número de canales de salida de cada capa es el número de entrada de la siguiente y los canales de salida de nuevo los estipulamos nosotros.
        El kernel_size esencialmente es el tamaño del filtro (Los píxeles que rodeen al central en la matriz de poderarán), elegimos el 3x3 porque es el más preciso y nuestras imágenes son pequeñas
        El padding se  aplica  para que el filtro no reduzca el tamaño de la imagen en cada paso (así los bordes también se analizan) Y al ser el kernel de 3x3, podemos poner pading solo de 1
        Se normalizan los canales de salida de cada capa con BatchNornm2d para que:
        -Acelere la convergencia
        -Evita explosiones o desapariciones de gradiente
        -Regulariza el modelo, reduciendo así la sensibilidad a cambios en los pesos o datos de entrada
        
        Maxpool2d va haciendo un pooling de la salida de cada capa dividiendo el tamaño de la imagen entre dos cada vez que se aplica una capa, si  la imagen es de 64x64 terminará siendo de 8x8
        '''
        
        # Definir las capas de la CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= out_1, kernel_size=3, padding=1)  #3 to 64, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn1 = nn.BatchNorm2d(out_1)  # Normaliza las salidas de conv1
        self.conv2 = nn.Conv2d(in_channels= out_1, out_channels= out_2, kernel_size=3, padding=1)  #64 to 128, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn2 = nn.BatchNorm2d(out_2)  # Normaliza las salidas de conv2
        self.conv3 = nn.Conv2d(in_channels=out_2, out_channels= out_3, kernel_size=3, padding=1)  #128 to 256, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn3 = nn.BatchNorm2d(out_3)  # Normaliza las salidas de conv3
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Kernel=2 indica que su ventana de pooling es de 2x2 y stride=2 indica que se moverá 2 píxeles evitando solapamiento
        
        '''
        Como tenemos 3 capas convolucionales y hemos aplicado después de cada una un maxpool(2,2), nos deja las imágenes con tamaño de 8x8 y 128 canales de salida al ser la salida de la última capa en términos de canales
        Elegimos 512 neuronas pero podemos cambiarlas, a menos si producen overfitting (sobreajuste) o a más si producen underfitting (subajuste)
        Elegimos utilizar la función de activación sigmoid en la capa final porque nuestro problema es binario, y convierte el valor de la neurona final en una probabilidad entre 0 y 1 (Si tuvieramos más de una clase habría tantas neuronas como clases y se usaría la función Softmax por ejemplo) 
        '''
        #FULLY CONNECTED LAYERS
        self.fc1 = nn.Linear(out_3 * self.image_size * self.image_size, 512)   # (256 * 8 * 8) 256 es el output de la última capa, y 8 * 8 son las dimensiones de la imagen al hacer un maxpool 3 veces (64x64 -> 8x8)
        self.fc2 = nn.Linear(512, 1)  # 512 neuronas y solo una neurona final a la que aplicaremos la función sigmoid para que deje una probabilidad normalizada entre 0 y 1 
        
        
    def forward(self, x):
        # Aplicar las capas de convolución, activaciones(Relu) y pooling 
        # Por orden, se aplica la capa convolucional, se normalizan las activaciones de la capa, se aplica la función Relu y se hace un pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (batch_size, 64, 64) -> (batch_size, 64, 32, 32)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (batch_size, 128, 32, 32) -> (batch_size, 128, 16, 16)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (batch_size, 256, 16, 16) -> (batch_size, 256, 8, 8)

        #FULLY CONNECTED LAYERS
        '''
        Al aplanar el tensor conseguimos que el tensor que teníamos antes [batch_size, 256, 8, 8] ([Batch_size, C, H, W]), aplicando x.view(-1, 256 * 8 * 8) se queda en [batch_size, 16384], es decir, deja la dimensión de batchsize y multiplica las otras dimensiones dejando un tensor [batch_size, 16384]
        '''
        # Aplanar el tensor 
        x = x.view(-1, self.out_3 * self.image_size * self.image_size)  # (batch_size, 256 * 8 * 8)
        
        # Capas totalmente conectadas
        x = F.relu(self.fc1(x))  # (batch_size, 512)
        x = F.dropout(x, p=0.5, training=self.training) # 50% de desactivación aleatoria 
        x = self.fc2(x)
        
        """
        Al aplicar la fórmula de pérdida BCE,  ya tiene incorporado una función sigmoid así que no hace falta aplicarla a la salida del modelo
        x = torch.sigmoid(self.fc2(x))  # (batch_size, 1) aplicamos la función de activación sigmoid para que nos de un valor normalizado entre 0 o 1 
        """

        return x

class CNN_4C(nn.Module):
   
    def __init__(self,out_1, out_2, out_3, out_4, image_size):
        super(CNN_4C, self).__init__()
        
        
        self.out_1 = out_1 #Salida 1, normalización y entrada 2
        self.out_2 = out_2 # Salida 2, normalización y entrada 3
        self.out_3 = out_3 # Salida 3, normalización y entrada 4
        self.out_4 = out_4 # Salida 3 y normalización
        self.image_size = int(image_size/ (2 ** 4)) #Tamaño nuevo después de 4 poolings
       
        
        # Definir las capas de la CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= out_1, kernel_size=3, padding=1)  #3 to 64, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn1 = nn.BatchNorm2d(out_1)  # Normaliza las salidas de conv1
        self.conv2 = nn.Conv2d(in_channels= out_1, out_channels= out_2, kernel_size=3, padding=1)  #64 to 128, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn2 = nn.BatchNorm2d(out_2)  # Normaliza las salidas de conv2
        self.conv3 = nn.Conv2d(in_channels=out_2, out_channels= out_3, kernel_size=3, padding=1)  #128 to 256, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn3 = nn.BatchNorm2d(out_3)  # Normaliza las salidas de conv3
        self.conv4 = nn.Conv2d(in_channels=out_3, out_channels= out_4, kernel_size=3, padding=1)  #256 to 512, kernel_size=3 == kernel_size=(3, 3), padding='same' daría como valor 1
        self.bn4 = nn.BatchNorm2d(out_4)  # Normaliza las salidas de conv4
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Kernel=2 indica que su ventana de pooling es de 2x2 y stride=2 indica que se moverá 2 píxeles evitando solapamiento
        
        
        #FULLY CONNECTED LAYERS
        self.fc1 = nn.Linear(out_4* self.image_size * self.image_size, 512)   # (512 * 4 * 4) 512 es el output de la última capa, y 4 * 4 son las dimensiones de la imagen al hacer un maxpool 4 veces (64x64 -> 4x4)
        self.fc2 = nn.Linear(512, 1)  # 512 neuronas y solo una neurona final a la que aplicaremos la función sigmoid para que deje una probabilidad normalizada entre 0 y 1 
        
        
    def forward(self, x):
        # Aplicar las capas de convolución, activaciones(Relu) y pooling 
        # Por orden, se aplica la capa convolucional, se normalizan las activaciones de la capa, se aplica la función Relu y se hace un pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (batch_size, 64, 64) -> (batch_size, 64, 32, 32)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (batch_size, 128, 32, 32) -> (batch_size, 128, 16, 16)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (batch_size, 256, 16, 16) -> (batch_size, 256, 8, 8)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # (batch_size, 512, 8, 8) -> (batch_size, 512, 4, 4) 
        #FULLY CONNECTED LAYERS
        
        # Aplanar el tensor 
        x = x.view(-1, self.out_4* self.image_size * self.image_size)  # (batch_size, 512 * 4 * 4)
        
        # Capas totalmente conectadas
        x = F.relu(self.fc1(x))  # (batch_size, 512)
        x = F.dropout(x, p=0.5, training=self.training) # 50% de desactivación aleatoria 
        x = self.fc2(x)
       

        return x
    
class CNN_B3C(nn.Module):
   
    def __init__(self,out_1, out_2, out_3, image_size):
        super(CNN_B3C, self).__init__()
        
        self.out_1 = out_1 #Salida 1, normalización y entrada 2
        self.out_2 = out_2 # Salida 2, normalización y entrada 3
        self.out_3 = out_3 # Salida 3 y normalización       
        self.image_size = int(image_size/ (2 ** 3)) #Tamaño nuevo después de 3 poolings
       
        
        # Definir las capas de la CNN
        #Primer bloque
        self.conv1a = nn.Conv2d(in_channels=3, out_channels= out_1, kernel_size=3, padding=1)  
        self.conv1b = nn.Conv2d(in_channels=out_1, out_channels= out_1, kernel_size=3, padding=1) 
        self.conv1c = nn.Conv2d(in_channels=out_1, out_channels= out_1, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm2d(out_1)  # Normaliza las salidas de conv1
        #Segundo bloque
        self.conv2a = nn.Conv2d(in_channels= out_1, out_channels= out_2, kernel_size=3, padding=1)  
        self.conv2b = nn.Conv2d(in_channels= out_2, out_channels= out_2, kernel_size=3, padding=1)  
        self.conv2c = nn.Conv2d(in_channels= out_2, out_channels= out_2, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm2d(out_2)  
        #Tercer bloque
        self.conv3a= nn.Conv2d(in_channels=out_2, out_channels= out_3, kernel_size=3, padding=1)  
        self.conv3b = nn.Conv2d(in_channels=out_3, out_channels= out_3, kernel_size=3, padding=1) 
        self.conv3c = nn.Conv2d(in_channels=out_3, out_channels= out_3, kernel_size=3, padding=1)  
        self.bn3 = nn.BatchNorm2d(out_3)  
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Kernel=2 indica que su ventana de pooling es de 2x2 y stride=2 indica que se moverá 2 píxeles evitando solapamiento
        
        
        #FULLY CONNECTED LAYERS
        self.fc1 = nn.Linear(out_3 * self.image_size * self.image_size, 512)   # (256 * 8 * 8) 256 es el output de la última capa, y 8 * 8 son las dimensiones de la imagen al hacer un maxpool 3 veces (64x64 -> 8x8)
        self.fc2 = nn.Linear(512, 1)  # 512 neuronas y solo una neurona final a la que aplicaremos la función sigmoid para que deje una probabilidad normalizada entre 0 y 1 
        
        
    def forward(self, x):
        # Aplicar las capas de convolución, activaciones(Relu) y pooling 
        # Por orden, se aplica la capa convolucional, se normalizan las activaciones de la capa, se aplica la función Relu y se hace un pool
        #Primer bloque
        x = F.relu(self.bn1(self.conv1a(x))) # (batch_size, 64, 64) -> (batch_size, 64, 32, 32)
        x = F.relu(self.bn1(self.conv1b(x))) # (batch_size, 64, 64) -> (batch_size, 64, 32, 32)
        x = self.pool(F.relu(self.bn1(self.conv1c(x)))) # (batch_size, 64, 64) -> (batch_size, 64, 32, 32)
        #Segundo bloque
        x = F.relu(self.bn2(self.conv2a(x))) # (batch_size, 128, 32, 32) -> (batch_size, 128, 16, 16)
        x = F.relu(self.bn2(self.conv2b(x))) # (batch_size, 128, 32, 32) -> (batch_size, 128, 16, 16)
        x = self.pool(F.relu(self.bn2(self.conv2c(x)))) # (batch_size, 128, 32, 32) -> (batch_size, 128, 16, 16)
        #Tercer bloque
        x = F.relu(self.bn3(self.conv3a(x))) # (batch_size, 256, 16, 16) -> (batch_size, 256, 8, 8)
        x = F.relu(self.bn3(self.conv3b(x))) # (batch_size, 256, 16, 16) -> (batch_size, 256, 8, 8)
        x = self.pool(F.relu(self.bn3(self.conv3c(x)))) # (batch_size, 256, 16, 16) -> (batch_size, 256, 8, 8)
        
        #FULLY CONNECTED LAYERS

        # Aplanar el tensor 
        x = x.view(-1, self.out_3 * self.image_size * self.image_size)  # (batch_size, 256 * 8 * 8)
        
        # Capas totalmente conectadas
        x = F.relu(self.fc1(x))  # (batch_size, 512)
        x = F.dropout(x, p=0.5, training=self.training) # 50% de desactivación aleatoria 
        x = self.fc2(x)
        

        return x
