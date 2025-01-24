import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms

import display as  disp

#ENTRENAMIENTO DEL MODELO
def entrenamiento_modelo(num_epochs, model, device, train_loader, validation_loader, learning_rate, batch_size, img_size, model_name):
    #Establecemos un optimizador
    '''
    Establecemos el optimizador Adam por ser uno de los más versátiles (pudiendo elegir SGD, RMSprop, Adagrad...) Dentro de todos ellos es el que mejor casa con nuestra red
    '''
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #Establecemos la función de pérdida CEWithLogitsLoss 
    '''
    Establecemos esta función porque en redes de clasificación binaria se recomienda esta opción, tenniendo solo una neurona en la capa final de las capas fullyconnected (Que determina una probabilidad en el rango de 0-1
    Se puede hacer también con la función CrossEntropyLoss dejando dos neuronas en la capa final (una para cada clase, en este caso FAKE o REAL)
    Esta función ya aplica la función sigmoid por dentro así que no hace falta aplicarla en la última capa de la red
    '''
    criterion = nn.BCEWithLogitsLoss()

    #Trazabilidad para la construcción de gráficas
    epochs = list(range(1, num_epochs+1))
    train_loss_values = []
    train_acc_values = []
    val_loss_values = []
    val_acc_values = []
    
    #Valores auxiliares para encontrar el mejor 
    best_val_acc = 0.0  #Guardamos la mejor precisión hasta el momento
    best_model_state = None  # Para guardar el estado del modelo del mejor epoch (Según validación)
    epoch_stop = 0 #Para guardar el mejor epoch
    aux_val_loss = 0.0 #Para guardar el valor de loss en validación
    aux_train_loss = 0.0 #Para guardar el valor de loss en entrenamiento
    aux_train_acc = 0.0 #Para guardar el valor de acc en entrenamiento
    save = False #Registro de guardados
    
    
    #ENTRENAMIENTO
    for epoch in range(num_epochs):
        model.train() #Ponemos el modelo en modo entrenamiento 
    
        train_running_loss = 0.0 # Acumular la pérdida del epoch
        train_running_accuracy = 0.0 #Acumular la precisión del epoch
    
    
        for i, batch in enumerate(train_loader):
            inputs, labels = batch # Obtener los datos de entrada y las etiquetas
        
            inputs = inputs.to(device) #Pasamos los imputs a la GPU para que pueda usarlos el modelo
            labels = labels.to(device) #Pasamos las labels a la GPU porque para la función de pérdida hacen falta en GPU porque realiza cálculos en el mismo dispositivo (GPU) que las salidas del modelo (outputs)
        
            labels = labels.float() # Transformar las etiquetas a float para la función de pérdida BCE que trabaja con floats
        
            optimizer.zero_grad() # Reiniciar los gradientes
        
            outputs = model(inputs) # Pasar las entradas por el modelo 
        
            loss = criterion(outputs.squeeze(), labels) # Calcular la pérdida
            acc = accuracy(outputs.squeeze(), labels) # Calcular la precisión
        
            loss.backward() # Retropropagación
            optimizer.step() # Actualizar los parámetros del modelo
        
            train_running_loss += loss.item() #Acumular pérdida
            train_running_accuracy += acc # Acumular precisión
        
        
            # Imprimir la pérdida por pasos
            #if i % 1000 == 0:  # Imprimir cada 1000 iteraciones
                #print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")
            
        #Imprimir la pérdida media y precisión del entrenamiento de cada epoch
        train_epoch_loss = train_running_loss / len(train_loader)
        train_epoch_accuracy = train_running_accuracy / len(train_loader)
        print("[ENTRENAMIENTO]")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {train_epoch_loss:.4f}, Accuracy: {train_epoch_accuracy:.2f}%")
        
        #Utilizamos los valores para graficar
        train_loss_values.append(train_epoch_loss) #Añadimos valor de entrenamiento para graficar
        train_acc_values.append(train_epoch_accuracy) #Añadimos valor de entrenamiento para graficar
    
        #Validación
        model.eval()  # Cambiamos el modelo a modo evaluación
        val_running_loss = 0.0 # Acumular la pérdida del epoch
        val_running_accuracy = 0.0 #Acumular la precisión del epoch
    
        with torch.no_grad(): #Como estamos en validación no calculamos gradientes
            for i, batch in enumerate(validation_loader):
                inputs, labels = batch # Obtener los datos de entrada y las etiquetas
        
                inputs = inputs.to(device) #Pasamos los imputs a la GPU para que pueda usarlos el modelo
                labels = labels.to(device) #Pasamos las labels a la GPU porque para la función de pérdida hacen falta en GPU porque realiza cálculos en el mismo dispositivo (GPU) que las salidas del modelo (outputs)
        
                labels = labels.float() # Transformar las etiquetas a float para la función de pérdida BCE que trabaja con floats
        
                outputs = model(inputs) # Pasar las entradas por el modelo 
        
                loss = criterion(outputs.squeeze(), labels) # Calcular la pérdida
                acc = accuracy(outputs.squeeze(), labels) # Calcular la precisión        
        
                val_running_loss += loss.item() #Acumular pérdida
                val_running_accuracy += acc # Acumular precisión
        
        
                # Imprimir la pérdida por pasos
                #if i % 1000 == 0:  # Imprimir cada 1000 iteraciones
                    #print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(validation_loader)}], Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%")
            
        #Imprimir la pérdida media y precisión de la validación de cada epoch
        val_epoch_loss = val_running_loss / len(validation_loader)
        val_epoch_accuracy = val_running_accuracy / len(validation_loader)
        
          #Guardamos los valores de la mejor etapa de validación
        if val_epoch_accuracy > best_val_acc:
            
            best_val_acc = val_epoch_accuracy #Actualizamos la mejor precisión
            best_model_state = model.state_dict() #Guardamos el estado del modelo en la mejor precisión en validación actual
            epoch_stop = epoch+1 
            aux_val_loss = val_epoch_loss
            aux_train_loss = train_epoch_loss 
            aux_train_acc = train_epoch_accuracy
            save = True
            
        print("[VALIDACIÓN]")
        if save == True:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.2f}% - Best")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.2f}%")
        
        save = False    
        
        #Utilizamos los valores para graficar
        val_loss_values.append(val_epoch_loss) #Añadimos valor de validación para graficar
        val_acc_values.append(val_epoch_accuracy) #Añadimos valor de validación para graficar
        
    #GUARDAMOS EL MODELO
    model_final_name = "Model" + model_name + ".pth" #Creamos el nombre del archivo del modelo
    model_path = "./Resultados/Models/" + model_final_name #Creamos la ruta del modelo
    torch.save(best_model_state, model_path) #Guardamos el modelo
    
    #GUARDAMOS EL ÚLTIMO EPOCH
    disp.guardar_hiperparámetros(aux_train_loss, aux_train_acc, aux_val_loss, best_val_acc, num_epochs, epoch_stop, learning_rate, batch_size, img_size, model_name)
    
    #GUARDAMOS LA GRÁFICA
    disp.guardar_graficas(epochs, train_loss_values, train_acc_values, val_loss_values, val_acc_values, model_name)
    
    #GUARDAMOS LA MATRIZ
    cm = disp.calcular_matriz_confusion(model, validation_loader, device, model_name)
    print("Matriz de Confusión:\n", cm)
    
    
    
#Establecemos la función de Accuracy
def accuracy(outputs, labels):
    #Aplicamos sigmoid para obtener probabilidades entre 0 y 1
    probabilidades = torch.sigmoid(outputs)
    #Convertimos las probabilidades a predicciones binarias con un umbral de 0.5 (mayor = 1, menor o igual = 0)
    predicciones = (probabilidades > 0.5).float()
    
    # Comparar las predicciones con las labels
    correctas = (predicciones == labels).sum().item()
    
    #Obtener el total
    total = labels.size(0)
    #Calcular la precisión
    accuracy = correctas / total * 100
    
    return accuracy

'''
Esta función mediante softmax se supone que no es ideal ya que Softmax está ideado para problemas de clasificación multiclase y Sigmoid es de clasificación binaria.
Tendría sentido usar Softmax si mi función de pérdida fuera CrossEntropyLoss que espera logits e internamente calcula Softmax y tendría sentido para mantener la consistencia
'''
def accuracy_softmax(outputs, labels):
    # Aplicamos Softmax para obtener probabilidades entre 0 y 1 para cada clase
    probabilidades = torch.softmax(outputs, dim=1)
    
    # Tomamos la clase con la probabilidad más alta como predicción
    predicciones = torch.argmax(probabilidades, dim=1)
    # Comparar las predicciones con las labels
    correctas = (predicciones == labels).sum().item()
    
    # Obtener el total
    total = labels.size(0)
    
    # Calcular la precisión
    accuracy = correctas / total * 100
    
    return accuracy