# CNN Image Classification LABB 1

### Image data has been removed to allow for reduced size before submission

#### Images can be downloaded from [flower recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)


Frågor:
• Motivera din modellarkitektur och val av relevanta hyperparametrar.
• Vilka aktiveringsfunktioner har du använt? Varför?
• Vilken loss funktion har du använt? Varför?
• Har du använt någon databehandling? Varför?
• Har du använt någon regulariseringsteknik? Motivera.
• Hur har modellen utvärderats?
• Är prestandan bra? Varför/ varför inte?
• Vad hade du kunnat göra för att förbättra den ytterligare?

1 - Motivera din modellarkitektur och val av relevanta hyperparametrar.

My model was build with the intent to search for the best parameters using a gridsearch. Due to little computational power available, this gridsearch was never completed.
The model comprises of 5 convolutional layers and 3 dense layers.
The first convolutional layer takes the input image with a defined shape, there is filter for each convolutional layer a kernel size, an activation function and padding set at "same". The same padding only ensures that the output size is same as the input size. The max pooling reduces the spatial dimension of the output, that is, it reduces the **height** **x** **width** of the output. The dropout works by randomly deactivating or turning off some neurons, that is setting their value to 0 and this helps reduce overfitting.
After all the convolutional networks, the flatten layer converts the 2D feature maps into 1D feature vector which are then passed into 2 fully connected dense layers and then into the final dense layer. The model is then compiled using the Adam optimizer and a loss function of categorical crossentropy. The model is trained using the fit method and the validation data is set to 20% of the training data.

2 - Vilka aktiveringsfunktioner har du använt? Varför?

The activation function used for the convolutional layers is the ReLU (Rectified Linear Unit) function. The ReLU function is used to add non-linearity to the network and is defined as f(x) = max(0,x). ReLU is linear for all positive values, and this means that it does not saturate. It will output the input directly if it is positive, otherwise, it will output zero.
The softmax activation function is used in the final layer to ensure that the output probabilities sum up to 1.

3 - Vilken loss funktion har du använt? Varför?
I used the categorical cross entropy since this is a classification problem. The goal of every machine learning problem is to minimize the loss function. The lower the loss function the better the model. In the case of deep learning where there are multiple outputs and I am using one hot encoding, the categorical cross entropy is the most desirable.

4 - Har du använt någon databehandling? Varför?
I set all the images to a shape of 224 \* 224, this is to ensure there is uniformity. converted each image into a tensor array and then normalised each of these images. Then image labels were also encoded such that the labels are between 0 and 4, for this model, i then used a onehot encoding such each label will for example have the shape [0,0,0,0,1] for the first label.

5 - Har du använt någon regulariseringsteknik? Motivera.
I used the l2 regularisation in the fully connected dense layers, that is dense layer 1 and 2. The L2 regularizer is also known as weight decay, that is, it aims to shrink the weights towards zero without ever making them zero.

6 - Hur har modellen utvärderats?
The model was evaluated using the accuracy metric.

7 - Är prestandan bra? Varför/ varför inte?
After many attempts and different parameter combinations, I dont think the model is as good as i would have liked it to be. I have had to choose the hyperparamters randomly and then test. And each iteration had a minimal change in accuracy. The accuracy was only improved by increasing the number of convolutional networks as well as number of epochs.
An obvious reason here could be that the data is overfitted since the accuracy is always sligtly better on the training data than on the validation data. This could be solved by doing data augmentation on the training data so that the model gets more images to learn from.
Due to computational power or lack thereof, I could have equally done a gridsearch to search for the best hyperameter and best combination of hyperparameters.
In a nutshell, a data augmentaion and a rigorous gridsearch.

8 - Vad hade du kunnat göra för att förbättra den ytterligare?

I could have used a grid search to find the best hyperparameters for the model.
Data augmentation to take the data from 4317 data points to maybe 100 000 data points.

## Transfer Learning

Frågor:
• Motivera ditt modellval
• Förklara hur du genomfört transfer learning.
• Förklara hur du systematiskt valt de bästa hyperparametrarna (ska framgå i koden).

1 - I used the Vgg19
VGG19 (Visual Geometry Group) is 19 layers deep and was trained on a dataset with more than a million images from the ImageNet database. It has 16 Convolutional layers and 3 fully connected layers. Given that my model had a small dataset issue, using a pretrained model improved the accuracy greatly. It still needs some improvements but it did much better.

2 - I created a base model from the VGG19 model but instantiating with imagenet as weights and the input shape being my **224 x 224** image shape
Then I added the VGG19 at the top of my sequence and serves as the foundational feature extractor. Flattened the network from 2d to 1d then added a fully connected network before having my output network.

3 - For tuning the model for best hyperparameters with the VGG19 pretrained ImageNet weights. include_top is set to False to excluse the fully connected layers, and the imput shape is set to **224 x 224**.
I choose to not train the first two layers of the network so that their pretrained weights are kept. I then expanded the model by adding a convolutional layer, a maxpooling and a dropout of 0.2 to reduce overfitting by randomly setting some neurons to 0. After flattening the network from 2D to 1D, th model is followed by two dense layers with 128 and 256 layers and both having ReLU activation functions. The model is compiled with categoricalcross-entropy and an adam optimizer with a learning rate of 0.001.
