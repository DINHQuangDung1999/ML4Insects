from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import preprocessing
from DataGenerator import get_loader

class Trainer():
    def __init__(self,X,y,model,criterion,optimizer,info):
        self.info = info
        self.model = model.to(self.info.device)
        self.criterion = criterion.to(self.info.device)
        self.optimizer = optimizer
        self.optimizer.param_groups[0]['lr'] = self.info.learning_rate
        self.result = {'training_loss': [],
                        'validation_accuracy': [], 
                        'class_accuracy': [], 
                        'test_score': [], 
                        'test_confusion_matrix': []}
        
        self.X, self.y = X, y

    def get_loader(self):
        print('Obtaning data loaders...')
        self.train_loader, self.validation_loader, self.test_loader = get_loader(self.X,self.y,batch_size=self.info.batch_size,model_type = self.info.model_type)
    
    def train_one_epoch(self):
        self.model.train()
        trainingloss = 0
        for i, (x_batch,y_batch) in enumerate(self.train_loader):    
            x_batch = x_batch.to(self.info.device)
            y_batch = y_batch.to(self.info.device)

            output = self.model(x_batch)

            loss = self.criterion(output,y_batch.ravel())
            trainingloss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        trainingloss = trainingloss/(i+1)
        self.result['training_loss'].append(trainingloss)

    def validate_one_epoch(self):
        self.model.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for x_batch,y_batch in self.validation_loader:
                x_batch = x_batch.to(self.info.device)
                y_batch = y_batch.to(self.info.device)

                outputs = self.model(x_batch)
                _,predicted = torch.max(outputs,1)

                n_samples += y_batch.size(0)
                n_correct += (predicted == y_batch.ravel()).sum().item()
            self.result['validation_accuracy'].append(n_correct/n_samples)

    def test(self, true_label):
        self.model.eval()
        with torch.no_grad():
            predicted_label = []
            for x_batch,y_batch in self.test_loader:
                
                x_batch = x_batch.to(self.info.device)
                y_batch = y_batch.to(self.info.device)

                outputs = self.model(x_batch)
                _,predicted = torch.max(outputs,1)

                predicted_label.append(predicted.cpu().numpy())
        
        predicted_label = np.concatenate([p for p in predicted_label])

        acc = round(accuracy_score(true_label,predicted_label),4)
        b_acc = round(balanced_accuracy_score(true_label,predicted_label),4)
        
        print(f'Accuracy : {acc}')
        print(f'Balanced accuracy: {b_acc}')
        
        print(f'Predicted labels: {np.unique(predicted_label,return_counts=True)}')
        print(f'True labels: {np.unique(true_label,return_counts=True)}')
        print('')

        c = confusion_matrix(true_label,predicted_label)
        n_class_predictions = np.sum(c,axis = 1)
        
        self.result['test_score'] = [acc,b_acc]
        self.result['test_class_accuracy'] = [round(c[i,i]/n_class_predictions[i],2) for i in range(len(n_class_predictions))]
        self.result['test_confusion_matrix'] = c

    def plot_result(self):
        _,((ax1,ax2,ax3,ax4)) = plt.subplots(1,4,figsize = (16,4))
        w = 0.2
        i = 0

        color = ['r','g','b','y']
        ax1_custom_legends = []

        ax1.plot(self.result['training_loss'], color[i], label = f'Training loss' )
        ax1.plot(self.result['validation_accuracy'], f'{color[i]}--', label = f'Validation acc')
        ax1_custom_legends.append(Line2D([0], [0], color=color[i], lw=4))

        h = self.result['test_class_accuracy']
        ax2.bar(len(h),h,w)

        h = self.result['test_score'][0]
        ax3.bar(len(h),h,w)
        i+=1
        
        ax1_custom_legends.append(Line2D([0], [0], color='black', linestyle = '--', lw=4))
        ax1_custom_legends.append(Line2D([0], [0], color='black', linestyle = '-', lw=4))
        ax1.set_xlabel('Epochs')
        ax1.set_title('Training loss and Validation accuracy')
        ax2.set_title('Class Accuracy')
        ax2.set_xticks(np.arange(0,7),preprocessing.labels_dict.keys())
        ax3.set_title('Test score')
        ax3.set_xticks(np.arange(0,2),['Accuracy','Balanced accuracy'])
        
        x = ConfusionMatrixDisplay(self.result['test_confusion_matrix'],display_labels=preprocessing.labels_dict.keys())
        x.plot(ax=ax4,colorbar = False)
        ax4.set_title('Confusion Matrix')

        ax1.legend()
        ax3.set_xticks(np.arange(0,2),labels = ['Accuracy','Balanced Accuracy'])

        plt.suptitle('Model performance',y = 1.1)
    
    def train(self, verbose = True):
        print('Initialize...')
        #Training 
        print('Training...')
        for epoch in range(self.info.num_epochs):
            
            self.train_one_epoch()
            self.validate_one_epoch()
            if verbose == True:
                if (epoch %10 == 0) or (epoch == self.info.num_epochs - 1):
                    print(f"Epoch [{epoch+1}/{self.info.num_epochs}], loss: {self.result['training_loss'][epoch]:.4f}, validation accuracy: {self.result['validation_accuracy'][epoch]:.4f}")  
        print('Testing...')
        #Testing
        self.test(true_label = self.y.label)
        print('Finished!')

    def save_checkpoint(self,name):
        torch.save(self.model,name)