





from autoencoder import latent_embed
from sklearn.neighbors import KDTree
import streamlit as st
from preprocessing import preprocessing, preprocessing_designer_data, original_designer_data

st.title('YUE QING WEI Recommender')

st.write("Don't ask me why, the machine said so")

new_pic=st.file_uploader('Upload a fashion item')
designer_data=preprocessing_designer_data()
designer_latent=latent_embed()
tree=KDTree(designer_latent)

images=original_designer_data()


if new_pic:
    new_data=preprocessing(new_pic)

    # st.write(new_data.shape)
    latent_new_data=latent_embed(new_data)
#     st.write(new_pic.shape)
#     st.write(new_data.shape)

    # st.write(f"{new_data[0]}")
    dis, ind=tree.query(latent_new_data,k=3)

    ind_pic=[*ind][0]



    st.image(images[ind_pic[0]])
    st.image(images[ind_pic[1]])
    st.image(images[ind_pic[2]])
