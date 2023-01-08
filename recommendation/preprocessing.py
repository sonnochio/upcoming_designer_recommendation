import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snsd
from PIL import Image
import os

#Getting directory
ROOT_PATH=os.getcwd()
PATH=os.path.join(ROOT_PATH,'images')
designer_image_files=os.listdir(PATH)

# bulk resizing designer images
def resizing(url):
    image=Image.open(url).resize((60,80))
    data=np.asarray(image, dtype="int32")
    return data

def preprocessing_designer_data():

    #bulk preprecessing designer images as numpy arrays
    data=[]
    for file in designer_image_files:
        url=os.path.join(PATH, file)
        data.append(resizing(url))

    designer_data=np.array(data)
    designer_data=designer_data.astype('float32')/255

    return designer_data

def original_designer_data():

    #bulk preprecessing designer images as numpy arrays
    images=[]
    for file in designer_image_files:
        url=os.path.join(PATH, file)
        image=Image.open(url)
        # data=np.asarray(image, dtype="int32")
        images.append(image)

    return images


def preprocessing(new_pic):
    image=Image.open(new_pic).resize((60,80))
    new_data=np.array(image, dtype="int32")
    new_data=np.array(new_data).astype('float32')/255

    new_data=np.expand_dims(new_data, axis=0)


    return new_data



# print(preprocessing()[0])
