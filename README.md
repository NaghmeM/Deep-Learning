# Fashion Classification Model with Deep Learning

## Project Overview

This project implements a Fashion Classification model using deep learning techniques to recognize different types of clothing items. The model is built using TensorFlow and Keras, leveraging transfer learning with the Xception architecture.

## Repository Contents

- `deep-learning.ipynb`: Jupyter notebook containing the full model development process.
- `clothing_model.tflite`: TensorFlow Lite model for clothing classification (output of the training process).
- `lambda_function.py`: Python script for deploying the model as an AWS Lambda function.
- `serverless.ipynb`: Jupyter notebook detailing the serverless deployment process.
- `test.py`: Script for testing the deployed model.
- `Dockerfile`: Configuration for creating a Docker image for Lambda deployment.

## Model Architecture

The model uses transfer learning with the Xception architecture:

```python
def make_model(input_size=150, learning_rate=0.01, size_inner=100, droprate=0.5):
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    outputs = keras.layers.Dense(10)(drop)
    
    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model
```

## Dataset

The model is trained on the Clothing Dataset Small Master, available at:
`/kaggle/input/clothing-dataset-small-master/clothing-dataset-small-master`

The dataset includes the following clothing categories:
['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

## Training Process

1. Data Preparation:
   - Uses `ImageDataGenerator` for data augmentation and preprocessing.
   - Applies transformations like shear, zoom, and horizontal flip to training data.

2. Model Training:
   - Utilizes transfer learning with Xception as the base model.
   - Trains for 50 epochs with model checkpointing to save the best model.
   - Uses Adam optimizer with a learning rate of 0.0005.

3. Model Evaluation:
   - Evaluates the trained model on a separate test dataset.

## Lambda Function for Deployment

The `lambda_function.py` file contains the code for deploying the model as an AWS Lambda function:

```python
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(299, 299))

interpreter = tflite.Interpreter(model_path='clothing_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

def predict(url):
    X = preprocessor.from_url(url)
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()
    return dict(zip(classes, float_predictions))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
```

## Docker Configuration

The Dockerfile for creating the Lambda deployment image:

```dockerfile
FROM public.ecr.aws/lambda/python:3.7

RUN pip3 install --upgrade pip

RUN pip3 install keras_image_helper --no-cache-dir
RUN pip3 install https://raw.githubusercontent.com/alexeygrigorev/serverless-deep-learning/master/tflite/tflite_runtime-2.2.0-cp37-cp37m-linux_x86_64.whl --no-cache-dir

COPY clothing_model.tflite clothing_model.tflite
COPY lambda_function.py lambda_function.py

CMD [ "lambda_function.lambda_handler" ]
```

## API Usage

The model can be used by sending a POST request to the deployed Lambda function URL with a JSON payload containing the 'url' key pointing to an image:

```json
{
  "url": "http://example.com/image-of-clothing.jpg"
}
```
 
The function will return a dictionary with class probabilities for the clothing item in the image.


## Contact

Najmeh Mohajeri - nmohajeri@gmail.com

