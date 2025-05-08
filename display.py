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
import numpy as np


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

    fig.savefig(os.path.join('Show', "Ejemplo_5_imagenes_DA.svg"), format='svg')   
    
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
    
    num_epochs = len(epochs) + 1
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
    ticks = np.arange(1, num_epochs, 1)
    plt.xticks(ticks, [int(x) for x in ticks]) # Formato a enteros
    plt.tight_layout()

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
    ticks = np.arange(1, num_epochs, 1)
    plt.xticks(ticks, [int(x) for x in ticks]) # Formato a enteros
    plt.tight_layout()
    
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
            plt.show()

    except Exception as e:
        print(f"Ocurrió un error al intentar guardar la matriz: {e}")

    return cm

def crear_nombre_modelo(arch, lr, da):
    #Creamos el nombre del modelo inicial
    if lr == 0.001:
        name = f"_{arch}_1_{da}"
    elif lr == 0.0001:
        name = f"_{arch}_2_{da}"
    elif lr == 0.0002:
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


def save_test_ConfusionMatrix(cm, model_name):
    
    # Creación
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'], annot_kws={"size": 16})
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    plt.title('Confusion Matrix')
    
    #Guardar matriz de confusión
    plt.tight_layout() 
    matriz_path = f'./Test/MatrixTest_{model_name}.svg'
    
    try:
        if os.path.exists(matriz_path):
            print(f"El archivo {matriz_path} ya existe, no se sobrescribirá.")
        else:
            plt.savefig(matriz_path, format='svg')
            print(f"Matriz guardada en {matriz_path}")
            plt.close()

    except Exception as e:
        print(f"Ocurrió un error al intentar guardar la matriz: {e}")
        
def save_test_results(model_name, acc, auc, tpr, fpr):
    
    #Truncamos el valor de auc
    rounded_auc = round(auc, 4)
    
    #Definimos la ruta a guardar
    csv_folder = 'Test'
    csv_file = os.path.join(csv_folder, 'Resultados_test.csv')
    
    # Datos a guardar
    data = [
        {
            "Model": model_name,
            "Accuracy (ACC)": acc,
            "Area Under the Curve (AUC)": rounded_auc,
            "True Positive Rate (TPR)": tpr,
            "False Positive Rate (FPR)": fpr
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

def save_test_results_prueba(model_name, acc, auc, tpr, fpr):
    
    #Truncamos el valor de auc
    rounded_auc = round(auc, 4)
    
    #Definimos la ruta a guardar
    csv_folder = 'Test'
    csv_file = os.path.join(csv_folder, 'Resultados_test_5_Thresholds.csv')
    
    # Datos a guardar
    data = [
        {
            "Model": model_name,
            "Accuracy (ACC)": acc,
            "Area Under the Curve (AUC)": rounded_auc,
            "True Positive Rate (TPR)": tpr,
            "False Positive Rate (FPR)": fpr
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
    
    
def save_roc(fpr_curve, tpr_curve, auc, tprs, fprs, auc_umbral, model_name):
    '''
    plt.plot(fpr_curve, tpr_curve, marker='.')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.grid(True)
    plt.show()
    '''
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_curve, tpr_curve, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    # Añadir línea diagonal (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  
    # Añadir el punto específico (umbral = 0.5)
    plt.plot(fprs, tprs, linewidth=0.7, color='red', label=f'5 Thresholds (AUC = {auc_umbral:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    roc_path = f'./Test/ROC_{model_name}.svg'
    
    try:
        if os.path.exists(roc_path):
            print(f"El archivo {roc_path} ya existe, no se sobrescribirá.")
        else:
            plt.savefig(roc_path, format='svg')
            print(f"Grafica ROC guardada en {roc_path}")
            plt.close()

    except Exception as e:
        print(f"Ocurrió un error al intentar guardar la grafica ROC: {e}")
        
        
        
def save_roc_three(tprs_3C, fprs_3C, auc_3C, tprs_4C, fprs_4C, auc_4C, tprs_B3C, fprs_B3C, auc_B3C):

    
    plt.figure(figsize=(8, 6))
    #Model3C
    plt.plot(fprs_3C, tprs_3C, color='darkorange', linewidth=1, label=f'ROC curve Model3C (AUC = {auc_3C:.3f})')
    #Model4C
    plt.plot(fprs_4C, tprs_4C, color='blue', linewidth=1, label=f'ROC curve Model4C (AUC = {auc_4C:.3f})')
    #ModelB3C
    plt.plot(fprs_B3C, tprs_B3C, color='red', linewidth=1, label=f'ROC curve ModelB3C (AUC = {auc_B3C:.3f})')
    # Añadir línea diagonal (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], color='0.8', lw=2, linestyle='--')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves Model3C - Model4C - ModelB3C')
    plt.legend(loc="lower right")
    
    roc_path = f'./Test/ROC_ALL_MODELS.svg'
    
    try:
        if os.path.exists(roc_path):
            print(f"El archivo {roc_path} ya existe, no se sobrescribirá.")
            plt.close()
        else:
            plt.savefig(roc_path, format='svg')
            print(f"Grafica ROC guardada en {roc_path}")
            plt.close()

    except Exception as e:
        print(f"Ocurrió un error al intentar guardar la grafica ROC: {e}")
    

    