import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def class_distribution_plt(train_df, column_name):
    
    '''Plots distribution of classes in the training dataframe and saves it as a jpg 
       in training_results folder
    
    Parameters
    ----------
    train_df : dataframe
        A dataframe of the training data
    column_name : string
        The name of the column that represents the label names

    '''

    sns.countplot(y=train_df[column_name])
    plt.title('Images per Sports',)
    plt.ylabel('Number of images')
    plt.xlabel('Sports Name')
    plt.tight_layout()
    plt.savefig('class_distribution.jpg')

def print_class_distribution(train_df, column_name):

    '''Prints the class distribution in console and saves it as a csv file 
       in training_results folder
    
    Parameters
    ----------
    train_df : dataframe
        A dataframe of the training data
    column_name : string
        The name of the column that represents the label names

    '''

    classes = train_df[column_name].unique()
    class_count = len(classes)
    print('The number of classes in the dataset is: ', class_count)
    groups=train_df.groupby(column_name)
    print('{0:^30s} {1:^13s}'.format('CLASS', 'IMAGE COUNT'))
    countlist=[]
    classlist=[]
    for label in train_df[column_name].unique():
        group=groups.get_group(label)
        countlist.append(len(group))
        classlist.append(label)
        print('{0:^30s} {1:^13s}'.format(label, str(len(group))))
    csv_output = np.column_stack((classlist,countlist))
    makefolder('training_results')
    np.savetxt('training_results/class_distribution.csv',csv_output, fmt='%s')
    max_value=np.max(countlist)
    max_index=countlist.index(max_value)
    max_class=classlist[max_index]
    min_value=np.min(countlist)
    min_index=countlist.index(min_value)
    min_class=classlist[min_index]
    print(max_class, ' has the most images= ',max_value, ' ', min_class, ' has the least images= ', min_value)

def tr_plot(tr_data, start_epoch):

    '''Plots Loss and Accuracy graphs for training and validation.
       saves it as jpg in training_results folder
       
    Parameters
    ----------
    tr_data : history object from model.fit
        History of all the training data accumulated during the training phase
    start_epoch : int
        The starting epoch of the plot

    '''

    tacc=tr_data.history['accuracy']
    tloss=tr_data.history['loss']
    vacc=tr_data.history['val_accuracy']
    vloss=tr_data.history['val_loss']
    Epoch_count=len(tacc)+ start_epoch
    Epochs=[]
    for i in range (start_epoch ,Epoch_count):
        Epochs.append(i+1)   
    index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss
    val_lowest=vloss[index_loss]
    index_acc=np.argmax(vacc)
    acc_highest=vacc[index_acc]
    plt.style.use('fivethirtyeight')
    sc_label='best epoch= '+ str(index_loss+1 +start_epoch)
    vc_label='best epoch= '+ str(index_acc + 1+ start_epoch)
    fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(20,8))
    axes[0].plot(Epochs,tloss, 'r', label='Training loss')
    axes[0].plot(Epochs,vloss,'g',label='Validation loss' )
    axes[0].scatter(index_loss+1 +start_epoch,val_lowest, s=150, c= 'blue', label=sc_label)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')
    axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')
    axes[1].scatter(index_acc+1 +start_epoch,acc_highest, s=150, c= 'blue', label=vc_label)
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout
    plt.savefig('training_results/train_and_val_graphs.jpg') 
    plt.show()

def makefolder(path):
    
    '''Makes a folder in a certain path if it doesn't exist already
       
    Parameters
    ----------
    path : string
        Path for the new folder
   
    '''
    
    # checking if the directory demo_folder 
    # exist or not.
    if not os.path.exists(path):
        
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(path)
