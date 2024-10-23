import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Pruebo con un batch_size de 32 de momento, no descarto subirlo dependiendo de la cantidad de imágenes que queramos usar
#Ya  que las  imágenes tienen una resolución de 32 x 32, voy a probar a dejar el tamaño de 32x32 o si no considero subirlo a 64x64

def get_data_loaders(data_dir, batch_size=32, img_size=32):
    """
    Carga las imágenes desde las carpetas REAL y FAKE (con el path work/dataset/train y el path work/dataset/test), aplica las transformaciones
    y retorna los DataLoader para entrenamiento y validación.
    """
    # Transformaciones: cambiar tamaño, rotar, hacer un flip horizontal y normalizar imágenes
     transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),       # Redimensionar a 32x32 en el caso de que la imagen no sea ya de ese tamaño
        transforms.RandomRotation(30),                    # Rotar aleatoriamente hasta 30 grados
        transforms.RandomHorizontalFlip(),         # Flip horizontal aleatorio (Se puede meter vertical también)
        transforms.ToTensor(),             # Convertir a tensor (rango de 0 a 1)
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  # Normalizar con valores medios y std típicos, siendo mean los valores probedios RGB de ImageNet 
        #transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))  # Normalización que ha usado Helena
    ])

    # Cargar las imágenes desde las carpetas con etiquetas automáticas (0 = FAKE, 1 = REAL)
    # Como ImageFolder se basa en el orden alfabético para la asignación de etiquetas, podemos saber el valor de cada una
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    
    # Crear DataLoader para entrenamiento
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader