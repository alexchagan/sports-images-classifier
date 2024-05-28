from keras.utils import image_dataset_from_directory

from utils import print_class_distribution, makefolder
import tensorflow as tf

import os
from utils import makefolder
import kaggle
import pandas as pd
import albumentations as A
import cv2
import shutil

import PIL
from pathlib import Path
from PIL import Image, UnidentifiedImageError

class DataHandler:

    #TODO: automatically determaine if balance_classes should be True based on the distribution of classes
    #TODO: change test_ds to test_df -> test_gen
    def __init__(self, batch_size, image_size, balance_classes):
        self.image_size = image_size
        self.batch_size = batch_size
        self.balance_classes = balance_classes
        self.train_df = None
        self.valid_df = None
        self.train_gen = None
        self.valid_gen = None
        self.test_ds = image_dataset_from_directory("./data/test", batch_size=self.batch_size, image_size=self.image_size, seed=56)


    @staticmethod
    def download_from_kaggle(user, dataset):
        
        '''Downloads data from kaggle and store is it in a local directory'''

        if not os.path.exists('.kaggle'):
            raise Exception('.kaggle file with kaggle credentials is mandatory')
        
        if os.path.exists('data'):
            print('Data already exists, no need to download again.')
            return
    
        kaggle.api.authenticate()
        makefolder('data')
        kaggle.api.dataset_download_files(f'{user}/{dataset}', path='data', unzip=True, force=True)
        DataHandler.convert_to_jpg('data/')
        
    @staticmethod
    def convert_to_jpg(path):
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            
            if os.path.isfile(item_path):
                _, extension = os.path.splitext(item)

                # Convert the file to JPG if it's not already a JPG
                if extension.lower() != ".jpg":
                    try:
                        
                        image = Image.open(item_path)
                        
                        base_name = os.path.splitext(item)[0]
                        
                        new_filename = base_name + ".jpg"
                        new_file_path = os.path.join(path, new_filename)
                        
                        image = image.convert("RGB")
                        image.save(new_file_path, "JPEG")
                        
                        print(f"Converted {item} to {new_filename}")
                    except IOError:
                        print(f"Skipped {item} (not an image file)")
            
            # If the item is a directory, recursively call the function
            elif os.path.isdir(item_path):
                DataHandler.convert_to_jpg(item_path)

    def prepare_datasets(self):
        class_df = pd.read_csv('data/sports.csv')
        self.train_df = class_df[class_df['data set'] == 'train']
        self.valid_df = class_df[class_df['data set'] == 'valid']

        self.train_df['filepaths'] = 'data/' + self.train_df['filepaths']
        self.valid_df['filepaths'] = 'data/' + self.valid_df['filepaths']
    
        if self.balance_classes:
            self.train_df = self.balance(n=200, column='labels', working_dir='data')
        
        # Find faulty images and remove them from dataframe

        for df in [self.train_df, self.valid_df]:
            faulty_images = []
            for _ , row in df.iterrows():
                img_path = row['filepaths']
                try:
                    _ = PIL.Image.open(img_path)
                except UnidentifiedImageError:
                    print(f"Faulty image detected: {img_path}")
                    faulty_images.append(img_path)

            df.drop(df[df['filepaths'].isin(faulty_images)].index, inplace=True)
            
    def define_generators(self):
        gen = tf.keras.preprocessing.image.ImageDataGenerator()
        ycol='labels'
        self.train_gen = gen.flow_from_dataframe(self.train_df, x_col='filepaths', y_col=ycol, target_size=self.image_size,seed=123,
                                   class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=self.batch_size, validate_filenames=False)
        
        self.valid_gen = gen.flow_from_dataframe(self.valid_df, x_col='filepaths', y_col=ycol, target_size=self.image_size,seed=123,
                                   class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=self.batch_size, validate_filenames=False)
        
        return self.train_gen, self.valid_gen

    def balance(self, n, column, working_dir):    
        def get_augmented_image(image): # given an image this function returns an augmented image
            width=int(image.shape[1]*.8)
            height=int(image.shape[0]*.8)
            transform= A.Compose([
                A.HorizontalFlip(p=.5),
                A.Rotate(limit=30, p=.25),
                A.RandomBrightnessContrast(p=.5),
                A.RandomGamma(p=.5),
                A.RandomCrop(width=width, height=height, p=.25) ])    
            return transform(image=image)['image']
       
        df = self.train_df.copy()
        
        print('Initial length of dataframe is ', len(df))
        aug_dir=os.path.join(working_dir, 'aug')# directory to store augmented images
        if os.path.isdir(aug_dir):# start with an empty directory
            shutil.rmtree(aug_dir)
        os.mkdir(aug_dir)        
        for label in df[column].unique():    
            dir_path=os.path.join(aug_dir,label)    
            os.mkdir(dir_path) # make class directories within aug directory
        # create and store the augmented images  
        total=0    
        groups=df.groupby(column) # group by class
        for label in df[column].unique():  # for every class
            print(f'augmenting images in train set  for class {label}')                                               
            group=groups.get_group(label)  # a dataframe holding only rows with the specified label 
            sample_count=len(group)   # determine how many samples there are in this class  
            if sample_count< n: # if the class has less than target number of images
                aug_img_count=0
                delta=n - sample_count  # number of augmented images to create
                target_dir=os.path.join(aug_dir, label)  # define where to write the images            
                print(f'augmenting class {label:25s}')
                for i in range(delta):                
                    j= i % sample_count # need this because we may have to go through the image list several times to get the needed number
                    img_path=group['filepaths'].iloc[j]
                    img=cv2.imread(img_path)
                    img=get_augmented_image(img)
                    fname=os.path.basename(img_path)
                    fname='aug' +str(i) +'-' +fname
                    dest_path=os.path.join(target_dir, fname)                
                    cv2.imwrite(dest_path, img)
                    aug_img_count +=1
                total +=aug_img_count
            
        print('')
        print('Total Augmented images created= ', total)
        # create aug_df and merge with train_df to create composite training set ndf
        aug_fpaths=[]
        aug_labels=[]
        classlist=sorted(os.listdir(aug_dir))
        for klass in classlist:
            classpath=os.path.join(aug_dir, klass)     
            flist=sorted(os.listdir(classpath))    
            for f in flist:        
                fpath=os.path.join(classpath,f)         
                aug_fpaths.append(fpath)
                aug_labels.append(klass)
        Fseries=pd.Series(aug_fpaths, name='filepaths')
        Lseries=pd.Series(aug_labels, name='labels')   
        aug_df=pd.concat([Fseries, Lseries], axis=1)         
        df=pd.concat([df,aug_df], axis=0).reset_index(drop=True)
        print('Length of augmented dataframe is now ', len(df))    
        return df 
    
    def get_test_dataset(self):
        return self.test_ds
    
    def print_distributions(self):
        print_class_distribution(self.train_df, 'labels')
        print_class_distribution(self.valid_df, 'labels')
        
if __name__ == '__main__':

    pass
    
   
