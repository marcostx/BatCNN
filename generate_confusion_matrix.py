# Classify test images
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import caffe
import pickle
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score


# Initialize transformers
def initialize_transformer():
  shape = (1, 3, 56, 92)
  transformer = caffe.io.Transformer({'data': shape})
  
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))

  return transformer

transformer_RGB = initialize_transformer()
train="bat_test2.txt"

file_ = open(train,'r')
lines = file_.readlines()
values=[]
y_true=[]
for val in lines:
  values.append("battrain/" + val.split(" ")[0])
  y_true.append(int(val.split(" ")[1]))

RGB_images = values

#classify images with singleFrame model
def classify_images(frames, net, transformer):

  input_images = []
  c=0
  output_predictions = np.zeros(len(frames))
  for im in frames:
    # reading the image
    input_im = caffe.io.load_image(im)

    input_im = caffe.io.resize_image(input_im, (56,92))
    caffe_in = transformer.preprocess('data',input_im)
    net.blobs['data'].data[...] = caffe_in
      
    out = net.forward()
    # getting the probabilities
    val =out['probs'][0][:8]

    output_predictions[c]=np.argmax(val)

    del input_im
    c+=1

  return output_predictions

def plot_confusion_matrix(cm, title='Confusion matrix Fold 0', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(8)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#Models and weights
model = 'deploy_mlpModel.prototxt'
RGB_singleFrame = 'mlp_model_fold_2_iter_800.caffemodel'

net =  caffe.Net(model, RGB_singleFrame, caffe.TEST)

output = classify_images(RGB_images, net, transformer_RGB)
del net

matrix = confusion_matrix(y_true,output)
precision = precision_score(y_true, output, average="weighted")
f1 = f1_score(y_true, output, average="weighted")
recall = recall_score(y_true, output, average="weighted")

print("precision : ", precision)
print("\n")
print("f1 : ", f1)
print("\n")
print("recall : ", recall)
print("\n")
print(matrix)