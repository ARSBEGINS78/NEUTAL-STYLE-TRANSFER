COMPANY:CODTECH IT SOLUTIONS 
NAME:ABDUL RAZZAK SHEIKH 
INTERNS J20: COD01234 
DOMAIN:ARTIFICIAL INTELLIGENCE 
DURATION:4 WEEKS 
MENTOR: NEELA SANTOSH 
##DESCRIPTION:i have used tensorflow module which is used for deep learning computations,numpy module which handles numerical operations and array manipulations,matplotlib.pyplot module which displays images,PIL.Image module which loads and processes images,tensorflow.keras.applications.vgg19 module which Imports the pre-trained VGG19 model; which extracts image features,tensorflow.keras.models.Model module which allows creation of a custom model for feature extraction,ipywidgets module which creates interactive UI elements for running style transfer and IPython.display.display which displays widgets in a Jupyter Notebook.
Loading and Preprocessing Images:
Opens the image using PIL.Image.open(image_path).
Converts the image to RGB format (img.convert('RGB')).
Resizes it to a fixed dimension (max_dim × max_dim).
Converts the image to a NumPy array with dtype=np.float32 for TensorFlow compatibility.
Expands dimensions (np.expand_dims) to match model input shape (1, height, width, channels).
Preprocesses the image using vgg19.preprocess_input(img), which normalizes pixel values for the VGG19 model.
Deprocessing the Image:
Reshapes the image to 512 × 512 × 3.
Adds back the mean pixel values used in vgg19.preprocess_input, effectively reversing the preprocessing step.
Converts the image from BGR to RGB (img[:, :, ::-1]) because VGG19 uses BGR.
Clips pixel values to the range [0, 255] and converts them to uint8 for display.
Loading the VGG19 Model:>
Loads a pre-trained VGG19 model without the fully connected layers (include_top=False).
Freezes the model’s weights (vgg.trainable = False) since we only need feature extraction, not training.
Computing Style and Content Loss:
Defines content layers: block5_conv2 is used to capture content information.
Defines style layers: multiple layers extract different levels of style information.
Assigns weights: higher content weight ensures the final image resembles the content image more.
Extracting Features:>
Creates a dictionary outputs that maps layer names to outputs.
Builds a feature extractor model.
Extracts the content image’s feature map from block5_conv2.
Extracts style features from multiple layers.
Computing Style Representation Using Gram Matrix:
The Gram matrix represents the style of an image by computing the correlations between feature maps.
The feature map is reshaped into a 2D tensor ([-1, channels]), and tf.matmul computes the Gram matrix.
Stores the Gram matrices for the style image.
Defining the Loss Function:
Computes content loss as the mean squared error between the generated and original content features.
Computes style loss as the sum of mean squared errors between the Gram matrices of the generated and style image.
Returns the total loss function.
Performing Style Transfer:>
Loads and preprocesses content and style images.
Builds the VGG19 model and computes the loss function.
Initializes generated_image as a trainable variable (starting from content image).
Uses Adam optimizer for gradient-based optimization.
Optimization Loop:>
