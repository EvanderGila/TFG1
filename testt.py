import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import architecture as arch
import display as disp


#Función para crear el modelo con la arquitectura adecuada para cada modelo 
def load_model(model_path, model_class, out_1, out_2, out_3, out_4, img_size):

    if model_class == "3C":
        model = arch.CNN_3C(out_1, out_2, out_3, img_size)
    elif model_class == "4C":
        model = arch.CNN_4C(out_1, out_2, out_3, out_4, img_size)
    elif model_class == "B3C":
        model = arch.CNN_B3C(out_1, out_2, out_3, img_size)
    
    #Hacemos esto de una forma separada porque para cambiar weights_only=True solo deja hacerlo así
    state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    
    #model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')), weights_only=True)
    model.eval()
    
    return model

def evaluate_model(model, test_loader, device, model_path):
    #Mandamos el modelo al GPU
    model.to(device)
    #Creamos las listas que contendrán las etiquetas y probabilidades
    all_labels, all_preds, all_probs = [], [], []
    
    probs_negative, probs_positive = [], []
    #Como estamos en test desactivamos el cálculo de gradientes
    with torch.no_grad():
        
        for inputs, labels in test_loader:
            #Mandamos los datos del TestLoader al dispositivo
            inputs, labels = inputs.to(device), labels.to(device)
            #Pasamos los datos al modelo y recogemos las salidas
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs)  #Sigmoid convierte logit en probabilidad
            preds = (probs >= 0.5).long()  # Umbral de 0.5 para clasificar como 0 (FAKE) o 1 (REAL)
            
            #Guardamos los resultados en las listas
            all_labels.extend(labels.cpu().numpy()) #Etiquetas "Verdaderas"
            all_preds.extend(preds.cpu().numpy())   #Predicciones del modelo
            all_probs.extend(probs.cpu().numpy())   #Probabilidades de la clase positiva
            
            for prob, label in zip(probs, labels):
                if label == 1:
                    probs_positive.append(prob.cpu().numpy().item())
                else:
                    probs_negative.append(prob.cpu().numpy().item())
            
    size_positive = len(probs_positive)
    size_negative = len(probs_negative)
    
    mean_positive = np.mean(probs_positive)
    std_positive = np.std(probs_positive)

    mean_negative = np.mean(probs_negative)
    std_negative = np.std(probs_negative)
    
    print(f"Tamaño Positivo: {size_positive}, Media Positiva: {mean_positive}, Std Positiva: {std_positive}")
    print(f"Tamaño Negativo: {size_negative}, Media Negativa: {mean_negative}, Std Negativa: {std_negative}")
    
    
    print("Min prob:", min(all_probs), "Max prob:", max(all_probs))
    print("Unique labels:", set(all_labels))
    
    #Cálculo de métricas
    acc = accuracy_score(all_labels, all_preds) #Calculamos la precisión mediante accuracy_score(), definida en la librería sklearn.metrics  
    #auc = roc_auc_score(all_labels, all_probs)  #Calculamos la auc mediante roc_auc_score(), definida en la librería sklearn.metrics
    cm = confusion_matrix(all_labels, all_preds)#Calculamos la matriz de confusión
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)  #Calculamos el True Positive Rate / Sensitivity (Recall)
    fpr = fp / (fp + tn)  #Calculamos el False Positive Rate
    
    # Cálculo de TPR y FPR para distintos umbrales
    fpr_curve, tpr_curve, thresholds = roc_curve(all_labels, all_probs)

    # Cálculo de AUC usando integración numérica (trapezoidal) en varios umbrales
    auc = np.trapz(tpr_curve, fpr_curve)
    
    # AUC manual usando aproximación de trapezoide entre (0,0) y (FPR, TPR) con umbral de 0.5
    auc_umbral = 0.5 * tpr * (1 - fpr) + 0.5 * tpr * fpr
    
    
    
    #Nos quedamos solo con el nombre del modelo
    model_name = os.path.basename(model_path)
    
    #Mostrar curva ROC -----------
    disp.show_roc(fpr_curve, tpr_curve, auc, tpr, fpr, auc_umbral, model_name)
    
    #Guardar la matriz de confusión en formato SVG
    disp.save_test_ConfusionMatrix(cm, model_name)
    
    #Guardamos en un CSV todos los datos
    disp.save_test_results_prueba(model_name, acc, auc, tpr, fpr)

    return {'ACC': acc, 'AUC': auc, 'TPR': tpr, 'FPR': fpr, 'AUC_umbral': auc_umbral}

def evaluate_model_roc(model, test_loader, device, model_path):
    #Mandamos el modelo al GPU
    model.to(device)
    #Creamos las listas que contendrán las etiquetas y probabilidades
    all_labels, all_preds, all_probs = [], [], []
    
    probs_negative, probs_positive = [], []
    #Como estamos en test desactivamos el cálculo de gradientes
    with torch.no_grad():
        
        for inputs, labels in test_loader:
            #Mandamos los datos del TestLoader al dispositivo
            inputs, labels = inputs.to(device), labels.to(device)
            #Pasamos los datos al modelo y recogemos las salidas
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs)  #Sigmoid convierte logit en probabilidad
            preds = (probs >= 0.5).long()  # Umbral de 0.5 para clasificar como 0 (FAKE) o 1 (REAL)
            
            #Guardamos los resultados en las listas
            all_labels.extend(labels.cpu().numpy()) #Etiquetas "Verdaderas"
            all_preds.extend(preds.cpu().numpy())   #Predicciones del modelo
            all_probs.extend(probs.cpu().numpy())   #Probabilidades de la clase positiva
            
            for prob, label in zip(probs, labels):
                if label == 1:
                    probs_positive.append(prob.cpu().numpy().item())
                else:
                    probs_negative.append(prob.cpu().numpy().item())
            
    size_positive = len(probs_positive)
    size_negative = len(probs_negative)
    
    mean_positive = np.mean(probs_positive)
    std_positive = np.std(probs_positive)

    mean_negative = np.mean(probs_negative)
    std_negative = np.std(probs_negative)
    
    print(f"Tamaño Positivo: {size_positive}, Media Positiva: {mean_positive}, Std Positiva: {std_positive}")
    print(f"Tamaño Negativo: {size_negative}, Media Negativa: {mean_negative}, Std Negativa: {std_negative}")
    
    
    print("Min prob:", min(all_probs), "Max prob:", max(all_probs))
    print("Unique labels:", set(all_labels))
    
    #Cálculo de métricas
    acc = accuracy_score(all_labels, all_preds) #Calculamos la precisión mediante accuracy_score(), definida en la librería sklearn.metrics  
    #auc = roc_auc_score(all_labels, all_probs)  #Calculamos la auc mediante roc_auc_score(), definida en la librería sklearn.metrics
    cm = confusion_matrix(all_labels, all_preds)#Calculamos la matriz de confusión
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)  #Calculamos el True Positive Rate / Sensitivity (Recall)
    fpr = fp / (fp + tn)  #Calculamos el False Positive Rate
    
    
    
    
    # Cálculo de TPR y FPR para 5 umbrales
    thresholds = np.linspace(0, 1, num=5) 
    fprs, tprs = calculate_roc_with_fixed_thresholds(all_labels, all_probs, thresholds)
    if fprs[0] > fprs[-1]:
        fprs = fprs[::-1]
        tprs = tprs[::-1]
    auc_umbral = np.trapz(tprs, fprs)
    
    
    # Cálculo de TPR y FPR para distintos umbrales
    fpr_curve, tpr_curve, thresholds = roc_curve(all_labels, all_probs)
    # Cálculo de AUC usando integración numérica (trapezoidal) en varios umbrales
    auc = np.trapz(tpr_curve, fpr_curve)
    
    '''
    #Para calcular el AUC de un solo umbral, en este caso, 0.5
    
    # AUC manual usando el área del triángulo con vértices (0,0), (FPR,TPR), (FPR,0) m´s el triángulo (FPR,1), (FPR, TPR), (1,1) con umbral de 0.5
    auc_umbral = 0.5 * tpr * fpr + 0.5 * tpr * (1 - fpr)
    '''
    
    
    #Nos quedamos solo con el nombre del modelo
    model_name = os.path.basename(model_path)
    
    #Mostrar curva ROC 
    disp.save_roc(fpr_curve, tpr_curve, auc, tprs, fprs, auc_umbral, model_name)
    
    #Guardar la matriz de confusión en formato SVG
    disp.save_test_ConfusionMatrix(cm, model_name)
    
    #Guardamos en un CSV todos los datos
    disp.save_test_results_prueba(model_name, acc, auc_umbral, tpr, fpr)

    return {'ACC': acc, 'AUC': auc, 'TPR': tpr, 'FPR': fpr, 'AUC_umbral': auc_umbral}

def calculate_roc_with_fixed_thresholds(labels, probs, thresholds):
    tprs = []
    fprs = []

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        tprs.append(tpr)
        fprs.append(fpr)

    return np.array(fprs), np.array(tprs)

def evaluate_model_roc_three(model_3C, model_4C, model_B3C, test_loader, device):
    
    #MODELO 3C
    model_3C.to(device)
    #Creamos las listas que contendrán las etiquetas y probabilidades
    all_labels, all_preds, all_probs = [], [], []
    #Como estamos en test desactivamos el cálculo de gradientes
    with torch.no_grad():
        for inputs, labels in test_loader:
            #Mandamos los datos del TestLoader al dispositivo
            inputs, labels = inputs.to(device), labels.to(device)
            #Pasamos los datos al modelo y recogemos las salidas
            outputs = model_3C(inputs)
            probs = torch.sigmoid(outputs)  #Sigmoid convierte logit en probabilidad
            preds = (probs >= 0.5).long()  # Umbral de 0.5 para clasificar como 0 (FAKE) o 1 (REAL)
            #Guardamos los resultados en las listas
            all_labels.extend(labels.cpu().numpy()) #Etiquetas "Verdaderas"
            all_preds.extend(preds.cpu().numpy())   #Predicciones del modelo
            all_probs.extend(probs.cpu().numpy())   #Probabilidades de la clase positiva
            
    cm = confusion_matrix(all_labels, all_preds)#Calculamos la matriz de confusión
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)  #Calculamos el True Positive Rate / Sensitivity (Recall)
    fpr = fp / (fp + tn)  #Calculamos el False Positive Rate
    # Cálculo de TPR y FPR para 5 umbrales
    thresholds = np.linspace(0, 1, num=5) 
    fprs_3C, tprs_3C = calculate_roc_with_fixed_thresholds(all_labels, all_probs, thresholds)
    if fprs_3C[0] > fprs_3C[-1]:
        fprs_3C = fprs_3C[::-1]
        tprs_3C = tprs_3C[::-1]
    auc_3C = np.trapz(tprs_3C, fprs_3C)
    
    #MODELO 4C
    model_4C.to(device)
    #Creamos las listas que contendrán las etiquetas y probabilidades
    all_labels, all_preds, all_probs = [], [], []
    #Como estamos en test desactivamos el cálculo de gradientes
    with torch.no_grad():
        for inputs, labels in test_loader:
            #Mandamos los datos del TestLoader al dispositivo
            inputs, labels = inputs.to(device), labels.to(device)
            #Pasamos los datos al modelo y recogemos las salidas
            outputs = model_4C(inputs)
            probs = torch.sigmoid(outputs)  #Sigmoid convierte logit en probabilidad
            preds = (probs >= 0.5).long()  # Umbral de 0.5 para clasificar como 0 (FAKE) o 1 (REAL)
            #Guardamos los resultados en las listas
            all_labels.extend(labels.cpu().numpy()) #Etiquetas "Verdaderas"
            all_preds.extend(preds.cpu().numpy())   #Predicciones del modelo
            all_probs.extend(probs.cpu().numpy())   #Probabilidades de la clase positiva
            
    cm = confusion_matrix(all_labels, all_preds)#Calculamos la matriz de confusión
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)  #Calculamos el True Positive Rate / Sensitivity (Recall)
    fpr = fp / (fp + tn)  #Calculamos el False Positive Rate
    # Cálculo de TPR y FPR para 5 umbrales
    thresholds = np.linspace(0, 1, num=5) 
    fprs_4C, tprs_4C = calculate_roc_with_fixed_thresholds(all_labels, all_probs, thresholds)
    if fprs_4C[0] > fprs_4C[-1]:
        fprs_4C = fprs_4C[::-1]
        tprs_4C = tprs_4C[::-1]
    auc_4C = np.trapz(tprs_4C, fprs_4C)
    
    #MODELO B3C
    model_B3C.to(device)
    #Creamos las listas que contendrán las etiquetas y probabilidades
    all_labels, all_preds, all_probs = [], [], []
    #Como estamos en test desactivamos el cálculo de gradientes
    with torch.no_grad():
        for inputs, labels in test_loader:
            #Mandamos los datos del TestLoader al dispositivo
            inputs, labels = inputs.to(device), labels.to(device)
            #Pasamos los datos al modelo y recogemos las salidas
            outputs = model_B3C(inputs)
            probs = torch.sigmoid(outputs)  #Sigmoid convierte logit en probabilidad
            preds = (probs >= 0.5).long()  # Umbral de 0.5 para clasificar como 0 (FAKE) o 1 (REAL)
            #Guardamos los resultados en las listas
            all_labels.extend(labels.cpu().numpy()) #Etiquetas "Verdaderas"
            all_preds.extend(preds.cpu().numpy())   #Predicciones del modelo
            all_probs.extend(probs.cpu().numpy())   #Probabilidades de la clase positiva
            
    cm = confusion_matrix(all_labels, all_preds)#Calculamos la matriz de confusión
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)  #Calculamos el True Positive Rate / Sensitivity (Recall)
    fpr = fp / (fp + tn)  #Calculamos el False Positive Rate
    # Cálculo de TPR y FPR para 5 umbrales
    thresholds = np.linspace(0, 1, num=5) 
    fprs_B3C, tprs_B3C = calculate_roc_with_fixed_thresholds(all_labels, all_probs, thresholds)
    if fprs_B3C[0] > fprs_B3C[-1]:
        fprs_B3C = fprs_B3C[::-1]
        tprs_B3C = tprs_B3C[::-1]
    auc_B3C = np.trapz(tprs_B3C, fprs_B3C)
    
    #Mostrar curva ROC 
    disp.save_roc_three(tprs_3C, fprs_3C, auc_3C, tprs_4C, fprs_4C, auc_4C, tprs_B3C, fprs_B3C, auc_B3C)
    
