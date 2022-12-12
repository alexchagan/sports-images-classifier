from google.cloud import storage
import os


path_to_credentials = './credentials/feisty-album-369005-086941277569.json'

def list_blobs(bucket_name):

    storage_client = storage.Client.from_service_account_json(path_to_credentials)

    blobs = storage_client.list_blobs(bucket_name)

    return blobs

def download_data_to_local_directory(bucket_name, local_dir):

    storage_client = storage.Client.from_service_account_json(path_to_credentials)
    blobs = storage_client.list_blobs(bucket_name)

    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    
    for blob in blobs:
        joined_path = os.path.join(local_dir, blob.name)

        if os.path.basename(joined_path) == '': # if the file is a folder
            if not os.path.isdir(joined_path): # check if the folder already exist
                os.makedirs(joined_path)
        
        else: # if the file is an image
            if not os.path.isfile(joined_path): # check if the file already exist
                if not os.path.isdir(os.path.dirname(joined_path)): # check if the folder of the class exist
                    os.makedirs(os.path.dirname(joined_path))
                blob.download_to_filename(joined_path)



if __name__ == '__main__':

    download_data_to_local_directory('sports-classifier-bucket', './sports-classifier-data')
    