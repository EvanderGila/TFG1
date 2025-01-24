import csv
import os
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def display_directory_images(exp_size, images_dir):
    if(os.path.basename(images_dir) == 'REAL'):
        print("IMÁGENES REALES")
    elif(os.path.basename(images_dir) == 'FAKE'):
        print("IMÁGENES FALSAS")
        
    img_inv_count = 0
    img_eq_count=0

    for image_name in os.listdir(images_dir):            #os.listdir(directorio) devuelve una lista con los nombres de todos los archivos del directorio
        image_path = os.path.join(images_dir, image_name)
        try:
            with Image.open(image_path) as img:
                if img.size != exp_size: 
                    img_inv_count += 1
                else:
                    img_eq_count += 1
        except Exception as e:
            print(f"Error al procesar la imagen {image_path}: {e}")
        
    print(f"El contador de imágenes con resolución distinta a 32x32 en la dirección {images_dir} es: {img_inv_count}")
    print(f"El contador de imágenes con resolución de 32x32 en la dirección {images_dir} es: {img_eq_count}")
    
def display_loader(data_loader,num):
    count = 0
    for images, labels in data_loader:
        print("Tamaño del batch de imágenes:", images.shape)  # [batch_size, C, H, W]
        print("Tamaño del batch de etiquetas:", labels.shape)  # [batch_size]
        print("Etiquetas del batch:", labels)
        count += 1
        if(count >=num):
            break
            
def display_labels_loader(data_loader):
    etiqueta_fake = 0
    etiqueta_real = 0
    
    for i, batch in enumerate(data_loader):
        
        inputs, labels = batch
        
        etiqueta_fake += (labels == 0).sum().item()
        etiqueta_real += (labels == 1).sum().item()
        
    print(f"Número de etiquetas fake= {etiqueta_fake}")
    print(f"Número de etiquetas reales= {etiqueta_real}")

            
            
def display_images(n_label, n_perlabel, images_desnormalizadas):
    fig, axes = plt.subplots(n_label, n_perlabel, figsize=(15, 10))
    rang= n_label * n_perlabel
    for i in range(rang):
        #Calculamos la fila con i // 5 y la columna con i % 5
        ax = axes[i // n_perlabel, i % n_perlabel]
        # Convertir el tensor de imagen a numpy para poder visualizarlo Tensor: (C,H,W)
        img = images_desnormalizadas[i].permute(1, 2, 0).numpy()  # Cambiar el orden a (H,W,C)
        ax.imshow(img)
        if (i // n_perlabel == 1):
            ax.set_title("REAL")
        else:
            ax.set_title("FAKE")
        
        #ax.axis('off') 
    
    plt.subplots_adjust(hspace=0.01)  # Reduce el valor de hspace para acercar las filas

    fig.savefig(os.path.join('Show', "Ejemplo_5_imagenes3.svg"), format='svg')   
    
    #Mostramos la figura
    plt.show()
    #Cerramos la figura
    plt.close()
    
def filter_images(n_perlabel, train_loader):
    
    images_fake = []
    images_real = []

    #Extraemos las imágenes falsas y reales
    for images, labels in train_loader:
    
        #Filtramos las imágenes para quedarnos con las falsas
        labels_cero = labels == 0 #Devuelve un tensor booleano
        images_cero = images[labels_cero] #Selecciona las imágenes que hemos encontrado con etiqueta 0
        
        #Filtramos las imágenes para quedarnos con las reales
        labels_uno = labels == 1 #Devuelve un tensor booleano
        images_uno = images[labels_uno]
        
        images_fake.extend(images_cero[:n_perlabel - len(images_fake)]) #Seleccionamos las imágenes falsas hasta 5, si no hay en el primer batch, se repite y calculamos la nueva cantidad (5,4,3,2,1 o 0)
        images_real.extend(images_uno[:n_perlabel - len(images_real)]) #Seleccionamos las imágenes reales hasta 5, si no hay en el primer batch, se repite y calculamos la nueva cantidad (5,4,3,2,1 o 0)
        
        #Si en todo el batch he conseguido 5 de cada, salgo del for, si no, pues no
        if len(images_fake) >= n_perlabel and len(images_real) >= n_perlabel:
            break
        
    #Convertimos estas  imágenes a tensores
    images_fake = torch.stack(images_fake) #(5, 3, 64, 64)
    images_real = torch.stack(images_real) #(5, 3, 64, 64)
    
    #Creamos un solo tensor (las primeras 5 son las fake y las últimas 5 son las real)
    images = torch.cat((images_fake, images_real), dim=0) #(10, 3, 64, 64)
    
    return images

def guardar_hiperparámetros(train_loss, train_acc, val_loss, val_acc, num_epochs, epoch_stop, learning_rate, batch_size, img_size, model_name):
    #Guardamos los hiperparámetros y los datos de pérdida y precisión:
    #Definimos la ruta
    csv_folder = 'Resultados'
    csv_file = os.path.join(csv_folder, 'Resultados.csv')
    
    #Preprocesamos los datos de pérdida y precisión
    #Entrenamiento
    rounded_train_loss = round(train_loss, 4)
    train_acc = train_acc / 100
    rounded_train_acc = round(train_acc, 4)
    #Validación
    rounded_val_loss = round(val_loss, 4)
    val_acc = val_acc / 100
    rounded_val_acc = round(val_acc, 4)
    #Nombre del modelo
    model_name = f"Model{model_name}"
    # Datos a guardar
    data = [
        {
            "Loss_Train": rounded_train_loss,
            "Accuracy_Train": rounded_train_acc,
            "Loss_Validation": rounded_val_loss,
            "Accuracy_Validation": rounded_val_acc,
            "Num_Epochs": num_epochs,
            "Epoch_Save": epoch_stop,
            "Learning_Rate": learning_rate,
            "Batch_Size": batch_size,
            "Img_Size": img_size,
            "Model": model_name
        }
    ]
    
    try:
        with open(csv_file, mode='a', newline='') as file:  # 'a' para añadir
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
    
            # Si el archivo no existe, escribir la cabecera
            if file.tell() == 0:
                writer.writeheader()

            # Añadir los datos
            writer.writerows(data)
    
            print(f"Resultados guardados en {csv_file}")
    except Exception as e:
        print(f"Error al guardar los resultados: {e}")
        
def guardar_graficas(epochs, train_loss_values, train_acc_values, val_loss_values, val_acc_values, model_name):
    
    # Crear la figura y los ejes
    plt.figure(figsize=(12, 6))

    # Gráfica del Loss
    plt.subplot(1, 2, 1) 
    plt.plot(epochs, train_loss_values, color='red', label='Train Loss')
    plt.plot(epochs, val_loss_values, color='orange', label='Validation Loss')
    # Encontrar el mínimo para train y validation
    train_min_idx = train_loss_values.index(min(train_loss_values))
    val_min_idx = val_loss_values.index(min(val_loss_values))
    #Marcar los mínimos
    plt.plot(epochs[train_min_idx], train_loss_values[train_min_idx], 'ro', label=f'Train Min: {train_loss_values[train_min_idx]:.2f}')
    plt.plot(epochs[val_min_idx], val_loss_values[val_min_idx], 'o', color='orange', label=f'Validation Min: {val_loss_values[val_min_idx]:.2f}')
    plt.title('Loss Graphic')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Gráfica del Accuracy
    plt.subplot(1, 2, 2)  
    plt.plot(epochs, train_acc_values, color='blue', label='Train Accuracy')
    plt.plot(epochs, val_acc_values, color='green', label='Validation Accuracy')
    # Encontrar el máximo para train y validation
    train_max_idx = train_acc_values.index(max(train_acc_values))
    val_max_idx = val_acc_values.index(max(val_acc_values))
    #Marcar los mínimos
    plt.plot(epochs[train_max_idx], train_acc_values[train_max_idx], 'bo', label=f'Train Max: {train_acc_values[train_max_idx]:.2f}')
    plt.plot(epochs[val_max_idx], val_acc_values[val_max_idx], 'go', label=f'Validation Max: {val_acc_values[val_max_idx]:.2f}')
    plt.title('Accuracy Graphic')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    '''
    #Conseguir la fila de csv:
    csv_file_path = './CSV/Resultados.csv'
    df = pd.read_csv(csv_file_path)
    numero_filas = len(df)
    '''
    #Guarda las gráficas
    plt.tight_layout() 
    grafica_path = f'./Resultados/Graphs/Graph{model_name}.svg'

    try:
        if os.path.exists(grafica_path):
            print(f"El archivo {grafica_path} ya existe, no se sobrescribirá.")
        else:
            plt.savefig(grafica_path, format='svg')
            print(f"Gráfica guardada en {grafica_path}")

    except Exception as e:
        print(f"Ocurrió un error al intentar guardar la gráfica: {e}")

def calcular_matriz_confusion(model, data_loader, device, model_name):
    model.eval()  # Modo evaluación
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Desactiva el cálculo de gradientes
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Predicciones
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)  #Como usamos BCEWithLogitsLoss
            preds = (probs > 0.5).long()  # Umbral de 0.5 para clasificar como 0 (FAKE) o 1 (REAL)

            # Acumula etiquetas reales y predicciones
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Cálculo de la matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)

    # Visualización
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    #Guardar matriz de confusión
    '''
    #Conseguir la fila de csv:
    csv_file_path = './CSV/Resultados.csv'
    df = pd.read_csv(csv_file_path)
    numero_filas = len(df)
    '''
    
    plt.tight_layout() 
    matriz_path = f'./Resultados/Matrices/Matrix{model_name}.svg'
    
    try:
        if os.path.exists(matriz_path):
            print(f"El archivo {matriz_path} ya existe, no se sobrescribirá.")
        else:
            plt.savefig(matriz_path, format='svg')
            print(f"Matriz guardada en {matriz_path}")

    except Exception as e:
        print(f"Ocurrió un error al intentar guardar la matriz: {e}")

    return cm

def crear_nombre_modelo(arch, lr, da):
    #Creamos el nombre del modelo inicial
    if lr == 0.01:
        name = f"_{arch}_1_{da}"
    elif lr == 0.001:
        name = f"_{arch}_2_{da}"
    elif lr == 0.0001:
        name = f"_{arch}_3_{da}"
    else:
        print("Error en el lr")
    #Ahora comprobamos que no haya ningún modelo con esta configuración, en el caso de haberlo, editamos el nombre de forma que siempre sea único
    #Para ello, utilizamos la carpeta de Graphs para comprobar que no haya un modelo igual, y en el caso de haberlo, sumamos uno a la versión
    aux_name = f"Graph1{name}.svg"
    aux_path = f"Resultados/Graphs/{aux_name}"
    # Si el archivo no existe, devolver la versión 1
    if not os.path.exists(aux_path):
        name = f"1{name}"
        return name
    # Si ya existe, buscar un número disponible
    version = 2
    while True:
        aux_name = f"Graph{version}{name}.svg"
        aux_path = os.path.join("Resultados/Graphs/", aux_name)
        if not os.path.exists(aux_path):
            return f"{version}{name}"
        version += 1
    