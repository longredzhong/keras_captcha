import numpy as np
from skimage import io
import os
from keras.models import load_model

def predict_img(model_path,img_data):
    model = load_model(model_path)
    pred = model.predict(img_data,batch_size = 50)
    outdict = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    f = open('mappings.txt','w')
    for i in range(pred.shape[0]):
        c0 = outdict[np.argmax(pred[i][:36])]
        c1 = outdict[np.argmax(pred[i][36:36*2])]
        c2 = outdict[np.argmax(pred[i][36*2:36*3])]
        c3 = outdict[np.argmax(pred[i][36*3:36*4])]
        c4 = outdict[np.argmax(pred[i][36*4:])]
        c = c0+c1+c2+c3+c4
        n = np.str("{:0>4d}".format(i))
        f.write(n+','+c+'\r')
    f.close()
    
def read_data_img(img_path):
    img_file = os.listdir(img_path)
    img_data = []
    for img in img_file:
        path = os.path.join(img_path,img)
        image = io.imread(path,0)
        img_data.append(image)
    img_data = np.array(img_data)
    img_data = img_data.reshape(img_data.shape[0], 60, 200, 1)
    img_data = img_data.astype('float32')
    img_data /= 255
    return img_data
    
def main():
    path = os.getcwd()
    model_path = os.path.join(path,'my_model_data_2.h5')
    img_path = os.path.join(path,'train_change')
    img_data = read_data_img(img_path)
    predict_img(model_path,img_data)
    
if __name__ == '__main__':
    main()