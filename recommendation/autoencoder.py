from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preprocessing import preprocessing_designer_data







def build_encoder(latent_dimension):
    encoder=Sequential()

    encoder.add(Conv2D(8,(3,3), input_shape=(80,60,3), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(16, (3, 3), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Conv2D(32, (3, 2), activation='relu'))
    encoder.add(MaxPooling2D(2))

    encoder.add(Flatten())
    encoder.add(Dense(latent_dimension, activation='sigmoid'))

    return encoder



def latent_embed(data=preprocessing_designer_data()):

    encoder=build_encoder(50)
    
    latent=encoder.predict(data)

    return latent
