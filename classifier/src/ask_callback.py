from tensorflow import keras
import time

# Implementation is taken from a kaggle notebook: https://www.kaggle.com/code/gpiosenka/custom-callback-to-continue-or-stop-training/notebook

class ASK(keras.callbacks.Callback):
    
    def __init__ (self, model, epochs,  ask_epoch): # initialization of the callback
        super(ASK, self).__init__()
        self._model = model               
        self._ask_epoch = ask_epoch
        self._epochs = epochs
        self._ask = True # if True query the user on a specified epoch
        
    def on_train_begin(self, logs=None): # this runs on the beginning of training
        if self._ask_epoch == 0: 
            print('you set ask_epoch = 0, ask_epoch will be set to 1', flush=True)
            self._ask_epoch=1
        if self._ask_epoch >= self._epochs: # you are running for epochs but ask_epoch>epochs
            print('ask_epoch >= epochs, will train for ', self._epochs, ' epochs', flush=True)
            self._ask=False # do not query the user
        if self._epochs == 1:
            self._ask=False # running only for 1 epoch so do not query user
        else:
            print('Training will proceed until epoch', self._ask_epoch,' then you will be asked to') 
            print(' enter H to halt training or enter an integer for how many more epochs to run then be asked again')  
        self.start_time= time.time() # set the time at which training started
        
    def on_train_end(self, logs=None):   # runs at the end of training     
        tr_duration=time.time() - self.start_time   # determine how long the training cycle lasted         
        hours = tr_duration // 3600
        minutes = (tr_duration - (hours * 3600)) // 60
        seconds = tr_duration - ((hours * 3600) + (minutes * 60))
        msg = f'training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds)'
        print (msg, flush=True) # print out training duration time
        
    def on_epoch_end(self, epoch, logs=None):  # method runs on the end of each epoch
        if self._ask: # are the conditions right to query the user?
            if epoch + 1 == self._ask_epoch: # is this epoch the one for quering the user?
                print('\n Enter H to end training or  an integer for the number of additional epochs to run then ask again')
                ans=input()
                
                if ans == 'H' or ans =='h' or ans == '0': # quit training for these conditions
                    print ('you entered ', ans, ' Training halted on epoch ', epoch+1, ' due to user input\n', flush=True)
                    self._model.stop_training = True # halt training
                else: # user wants to continue training
                    self._ask_epoch += int(ans)
                    if self._ask_epoch > self._epochs:
                        print('\nYou specified maximum epochs of as ', self._epochs, ' cannot train for ', self._ask_epoch, flush =True)
                    else:
                        print ('you entered ', ans, ' Training will continue to epoch ', self._ask_epoch, flush=True)