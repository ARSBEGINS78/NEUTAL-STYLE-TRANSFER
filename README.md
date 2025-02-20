COMPANY:CODTECH IT SOLUTIONS 
NAME:ABDUL RAZZAK SHEIKH 
INTERNS J20: COD01234 
DOMAIN:ARTIFICIAL INTELLIGENCE 
DURATION:4 WEEKS 
MENTOR: NEELA SANTOSH 
##DESCRIPTION:tensorflow: Provides deep learning functionalities.
numpy: Used for numerical operations and handling image data.
matplotlib.pyplot: Used to display the final output image.
PIL.Image: Helps in loading and processing images.
vgg19: A pre-trained deep learning model used for feature extraction.
Model: A Keras class that allows us to create and manipulate models.
Opens an image file.
Converts it to RGB to ensure color consistency.
Resizes it to 512×512 (or another fixed size).
Converts the image to a NumPy array for numerical operations.
Adds a batch dimension since deep learning models expect batches.
Uses vgg19.preprocess_input() to normalize pixel values by subtracting the mean of the ImageNet dataset.
Converts the processed image back to a normal image.
Restores original color values by adding back the ImageNet mean pixel values.
Swaps color channels from BGR → RGB (VGG19 works with BGR).
Clips values to [0, 255] to ensure valid pixel intensities.
Converts the image to uint8 format for display.
Loads the VGG19 model pre-trained on ImageNet.
include_top=False ensures we only use convolutional layers (without classification layers).
Freezes the model since we only need it for feature extraction, not training.
Defines content layers (which hold structure) and style layers (which hold texture).
Extracts features using the VGG19 model.
Computes the Gram matrix, which captures style patterns.
Defines content loss (difference between content features) and style loss (difference between Gram matrices).
Returns a loss function to be used in optimization.
Loads and processes content and style images.
Initializes generated_image as a trainable variable.
Uses Adam optimizer to update the image.
Defines train_step() function using TensorFlow's automatic differentiation (GradientTape).
Runs iterations of gradient descent to minimize the loss.
Every 100 iterations, prints the loss value.
Displays the final stylized image.
Takes content and style image paths from the user.
Calls style_transfer() to perform Neural Style Transfer.
