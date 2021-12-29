# Kaggle-Competition
## Competition name: The Nature Conservancy Fisheries Monitoring
Link: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/overview

# Training
'train_regnet.ipynb'
This code can run on Colab, you may need to modify some data paths if you want to run it on your own machine.
Some inputs for training: 
- 'train.zip': training images downloaded from kaggle.
- 'label.zip': bounding boxes annotation using labelme (https://github.com/wkentaro/labelme).
- 'train_data.npz': a list of training file paths and the corresponding labels.

# Inference
model download link (regnet_y_8gf) : 
https://drive.google.com/file/d/1dnixFDB08Ps30clERNUJ6VSHzxS3zEpf/view?usp=sharing

- 'inference.py': resize and normalize the testing images downloaded from kaggle, and output the predictions from the model.
- 'inference_argument_regnet.py': implement the testing augmentation. The final result is the average of five predictions: 4 from random augmented data and 1 from original data.

Output: a csv file with the predictions from the model.
