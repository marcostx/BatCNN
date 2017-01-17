# BatCNN

For train the network, run this command:
## ./run_model.sh

evaluate the model, running :
## python generate_confusion_matrix.py

If you want to see the graph of accuracy x iterations, run:
## python generate_metrics.py

The cnn network model is defined in CNN.prototxt . The mlp model can be run (training and test) with :
## python mlp.py path_to_images

