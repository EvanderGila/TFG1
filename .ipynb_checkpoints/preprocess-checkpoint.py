import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import RandAugment



def get_train_loader(data_dir, batch_size=32, img_size=64):
    '''
        Como el transform.resize puede cambiar la media y la desviación, es mejor hacerlo antes de calcularlas. En el caso de cambiar a otro tamaño, se debería VOLVER A CALCULAR mean y std.
        El transfor.TorTensor transforma la imagen en un tensor, escala los valores de 0-255 dividiendo entre 255 para conseguir un rango [0-1], dejando el siguiente formato: Imagen: (canales, alto, ancho)
        De tal forma que tendrías todos los valores de los píxeles (alto y largo) de los 3 canales (RGB) con valores entre el 0 y el 1.
        El transform.Normalize normaliza los valores de los canales de la imagen mediante unos valores de mean y std, siendo el valor_normalizado= (valor - media) / desviacion_estandar.  Centrando así la media en         0 y dejando una desviación típica unitaria (los valores están distribuidos de forma estandar)
        
    '''

    #Transformamos las imágens con esta función de transformación (No utilizamos data augmentation porque tenemos ya suficientes muestras en el dataset)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), #1. Transformamos todas las imágenes al tamaño deseado (En este caso 64x64, ya que todas son 32x32 y pueden quedarse pequeñas)        
        transforms.ToTensor(), #2. Transforma las imágenes a tensores
        transforms.Normalize(mean=[0.4718, 0.4628, 0.4176],std=[0.2361, 0.2360, 0.2636])  
        # Normalizar con valores medios y std, los valores que he encontrado en internet son estos: mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
     
        
    ])
    '''
    El datasets.ImageFolder devuelve un objeto Dataset donde cada elemento es una tupla (image, label)
    Este train dataset contiene:
    - Las imágenes en formato tensor y sus transformaciones especificadas (Resize, Tensor y normalización)
    - Las etiquetas correspondientes con la jerarquía de archivos 0 FAKE 1 REAL (Se asigna 0 al primer directorio por orden alfabético el resto)
    image, label = dataset[X]
    Image es la imagen X del conjunto de datos procesada ya por las transformaciones, guardada como un tensor con la forma (3,H,W), siendo 3 el número de canales (RGB), H (height) y W (Width)
    Label es la etiqueta de la imagen X, pudiendo ser 0 (fake) o 1 (real), siendo una lista de ENTEROS
    '''
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train', 'training'), transform=transform)
    
    '''
    Este train loader es un tensor que agrupa imágenes en paquetes/lotes (Batches), contiene (Usando la función collate_fn, o por defecto):
    -images: Tensor que tiene forma (batch_size,3,H,W), siendo batch_size el número de imágenes. Forma de las imágenes: torch.Size([32, 3, 64, 64])
    -labels: Tensor de tipo LongTensor que contiene las  etiquetas de clase de sus imágenes [1, 0, 0, 1...] tantas etiquetas como batch_size haya (Una para cada imagen).  Forma de las etiquetas: torch.Size([32])
    DataLoader devuelve un iterador para recorrer  esas duplas de tensores de tamaño batch
    '''
    #COLLATE Y TORCH.STACK
    '''
    Trabajamos con dos tensores diferenciados para entrenar el modelo tal y como he visto en ejemplos (X_training, X_val, y_training, y_val del modelo 102 de general models por ejemplo)
    ¿¿¿Para lograrlo, modificamos la función collate_default que devuelve una dupla de tamaño batch de muestras y labels para que devuelva por un lado un tensor de muestras y otro de labels divididos por batches?????
    '''
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_custom)
    

    return train_loader

def get_train_loader_da(data_dir, batch_size=32, img_size=64):

    #Transformamos las imágens con esta función de transformación (No utilizamos data augmentation porque tenemos ya suficientes muestras en el dataset)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), #1. Transformamos todas las imágenes al tamaño deseado (En este caso 64x64, ya que todas son 32x32 y pueden quedarse pequeñas) 
        RandAugment(num_ops=2, magnitude=9), #2. Aplicamos Data Augmentation mediante RandAugment (num_operaciones, magnitud/intensidad de 0 a 30 siendo 30 la mayor)
        transforms.ToTensor(), #3. Transforma las imágenes a tensores
        transforms.Normalize(mean=[0.4718, 0.4628, 0.4176],std=[0.2361, 0.2360, 0.2636])  
        # Normalizar con valores medios y std, los valores que he encontrado en internet son estos: mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]
     
        
    ])
 
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train', 'training'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_custom)

    return train_loader
    
def collate_custom(batch):
    """
    Esta función toma una lista de muestras y las combina en un solo tensor por cada tipo (imágenes, etiquetas).
    batch: Lista de tuplas (imagen, etiqueta), siendo el train_dataset/validation_dataset esta lista
    """
    # Separar imágenes y etiquetas de la lista de tuplas
    images = [item[0] for item in batch]  # Extraer todas las imágenes
    labels = [item[1] for item in batch]  # Extraer todas las etiquetas
    
    
    images_tensor = torch.stack(images)  # Apilar todas las imágenes en un solo tensor, añadiendo en la  dim=0 el batch_size
    
    
    labels_tensor = torch.tensor(labels)  # Convertir las etiquetas a tensor (para hacer torch.stack(labels) tendrían que ser tensores antes, y son ENTEROS) Podría convertirlos debajo
    #labels_tensor = torch.stack(labels)
    
    return images_tensor, labels_tensor
    

'''
La validación se busca detectar el sobreajuste (Cuando el moddelo solo funciona con los datos de entrenamiento y no con datos nuevos, es decir, no generaliza)
Vale para monitorizar el rendimiento a lo largo el entrenamiento y ayuda a decidir cuándo detener el entrenamiento (early stopping) a parte de ajustar los hiperparámetros
En este caso, utilizaremos el directorio ./dataset/train/validation/REAL y ./dataset/train/validation/FAKE para alojar las imágenes para la validación con 15000 imágenes cada una de 32x32
Hago un resize a 64x64 y normalizo con los datos de la media y std del conjunto de TRAINING.
Devolvemos, como en el training, dos tensores, uno con las imágenes y otro con las etiquetas

'''
def get_validation_loader(data_dir, batch_size=32, img_size=64):
    
     #Transformamos las imágenes en un tensor, antes reescalando las imágenes a 64x64 porque el entrenamiento se ha hecho así, también las normalizamos con la media de los datos del training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),       
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4718, 0.4628, 0.4176],std=[0.2361, 0.2360, 0.2636])  #Datos de mean y std del conjunto de TRAINING
        
    ])
    
    validation_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train', 'validation'), transform=transform)
    
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
  
    return validation_loader


'''
En la documentación que me ha pasado Helena 
def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
        std += images.std(axis=(0, 2, 3)).sum()

    mean /= num_pixels
    std /= num_pixels

    return mean, std
'''


def get_media_std_training(data_dir):
    
    #Solo lo convertimos a tensor para calcular los valores de la media y la desviación de las imágenes SIN ALTERAR
    transform = transforms.Compose([
        transforms.ToTensor()  
    ])
    #Creamos el dataset con las imágenes "raw" como tensor
    calculo_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train', 'training'), transform=transform)
    
    #Vamos cargando los datos en el loader de 64 en 64 para después utilizarlo (De 64 en 64 en vez de 32 en 32 porque es más rápido
    loader = DataLoader(calculo_dataset, batch_size=64, shuffle=False)
    
    
    mean = torch.zeros(3)  #Tensor de tamaño 3 para almacenar la media de cada canal RGB
    std = torch.zeros(3)   #Tensor para almacenar la desviación estándar para cada canal RGB
    num_muestras = 0  # Contador del número total de muestras para dividir por él y para comprobar que se hace bien
    
    #Del data loader cogemos el tensor de las imágenes de la tupla que nos proporciona y dejamos en blanco las labels, esto con el torch.stack se podrían pasar solo las images y no tendríamos que hacer lo de dejar en blanco las labels imagino 
    for images, _ in loader:
      
        #Acumular la media de los tres canales
        mean += images.mean([0, 2, 3]) * images.size(0)  # Sumar la media de cada canal "images.mean([0, 2, 3])" ya que la posición 1 del tensor images son los canales RGB (con valor 3). Luego multiplicar         por el número de imágenes (tamaño del batch)
        std += images.std([0, 2, 3]) * images.size(0)  # Sumar la desviación estandar de cada canal "images.mean([0, 2, 3])" ya que la posición 1 del tensor son los canales RGB (con valor 3). Luego         multiplicar por el número de imágenes (tamaño del batch)
    
        num_muestras += images.size(0)  # Acumular el número total de muestras, en este caso tendría que ser 70000 (35000 de FAKE y 35000 de REAL)

    if num_muestras == 70000:
        # Calcular la media y desviación estándar global
        mean /= num_muestras
        std /= num_muestras
    
        print("Media:", mean)
        print("Desviación estándar:", std)
    
    else:
        
        print(f"Error, no hay 70000 muestras, hay:{num_muestras}")
  #También creo que se podría haber calculado sin tener que muktiplicar en cada iteración del for por el images.size(0), simplemente llevando un contador en vez de del num_muestras, de el número de iteraciones (pero habría que tener controlado que la última iteración puede que el batch no estuviera completo y sería impreciso), y en vez de dividir por el num_muestras divides por el num_iteraciones al final
def get_test_loader(data_dir, batch_size=32, img_size=64):
    
     #Transformamos las imágenes en un tensor, antes reescalando las imágenes a 64x64 porque el entrenamiento se ha hecho así, también las normalizamos con la media de los datos del training
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),       
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4718, 0.4628, 0.4176],std=[0.2361, 0.2360, 0.2636])  #Datos de mean y std del conjunto de TRAINING
        
    ])
    
    test_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) #Dejamos suffle a false porque estamos en test y no hace falta
  
    return test_loader