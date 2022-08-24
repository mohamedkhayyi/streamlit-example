mport streamlit as st
import cv2
import numpy as np  
import os
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Input
import json
import os.path

path=r"C:\Users\mohamed\Desktop\Stage d'applicattion"
print(os.path.exists(r"C:\Users\mohamed\Desktop\Stage d'applicattion\aa.jpg"))

def load_image(image_file):
    img = Image.open(image_file)
    return img 

st.subheader("Home")
image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    st.write(file_details)
    img = load_image(image_file)
    st.image(img)
    with open(os.path.join("images",image_file.name),"wb") as f:
       f.write((image_file).getbuffer())        
    st.success("Image Saved")
    img_array=cv2.imread(os.path.join("images",image_file.name))
    imgHSV=cv2.cvtColor(img_array,cv2.COLOR_RGB2HSV)
    ll=np.array([0,194,102])
    ul=np.array([68,255,255])
    mask=cv2.inRange(imgHSV,ll,ul)
    cv2.imwrite('mask.jpg',mask)
    st.image(mask)
    st.success("Image Processed")
    #imgs =Image.open(img)
    masks =Image.open('mask.jpg')

    # Normalisation ==> [image] / 255
    I = img.getdata()
    L = masks.getdata()
    img_array = np.array(I)/255
    label_array = np.array(L)/255
    s = img_array
    t = label_array
    # Creation du modele 3-3-1
    model = Sequential([
     Input(3),
     Dense( 3, activation='sigmoid'),
     Dense( 1, activation='sigmoid')
     
     ])
    # Compilation
    model.compile(loss='mean_squared_error', optimizer='sgd',
    metrics=['accuracy'])
    # Affichage du resume
    model.summary()
    # Entrainement du modele
    history = model.fit(s, t, epochs=10, batch_size=10)
    # Recuperation des statistiques
    imgT =Image.open(r"C:\Users\mohamed\Desktop\Stage d'applicattion\bb.jpg")
    maskT=Image.open(r"C:\Users\mohamed\Desktop\Stage d'applicattion\dd.png")
    K = imgT.getdata()
    J = maskT.getdata()
    img_array = np.array(K)/255
    label_array = np.array(J)/255
    x = img_array
    y = label_array
    (val_loss,val_accuracy) = model.evaluate(x,y)
    score = model.evaluate(x, y, verbose=0)
    print('Test loss :', score[0])
    print('Test accuracy :', score[1])
    #Recuperation des poids du reseau
    weights_1 = model.layers[1].get_weights()[0].tolist()
    biases_1 = model.layers[1].get_weights()[1].tolist()
    data = {"weights": weights_1,"biases":biases_1}
    f = open('weightsandbiases.txt', "w")
    json.dump(data, f)
    f.close()
    st.image(imgT)
    st.image(maskT)
    st.success("validation image")
    st.write(data)
    st.write('Test loss :', score[0])
    st.write('Test accuracy :', score[1])
