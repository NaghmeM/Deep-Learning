
# Fashion Classification Model with AWS Lambda Deployment

## Project Overview

This project implements a Fashion Classification model using deep learning techniques to demonstrate how neural networks work. The model is trained to recognize different types of clothing items and is deployed as a serverless application using AWS Lambda.

## Repository Contents

- `clothing_model.tflite`: TensorFlow Lite model for clothing classification.
- `deep-learning.ipynb`: Jupyter notebook containing the deep learning model development process.
- `lambda_function.py`: Python script for the AWS Lambda function.
- `serverless.ipynb`: Jupyter notebook detailing the serverless deployment process.
- `test.py`: Script for testing the Lambda function.
- `xception_v4_1_epoch_11_val_accuracy_0.889.h5`: Trained Xception model with validation accuracy of 88.9%.
- `Dockerfile`: Configuration for creating a Docker image for Lambda deployment.

## Model Architecture

The model uses a transfer learning approach with the Xception architecture:

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

## Lambda Function

The `lambda_function.py` file contains the code for the AWS Lambda function that utilizes the TFLite model:

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

## Dockerfile

The Dockerfile is used to create a container image for deploying the model on AWS Lambda:

```dockerfile
FROM public.ecr.aws/lambda/python:3.7

RUN pip3 install --upgrade pip

RUN pip3 install keras_image_helper --no-cache-dir
RUN pip3 install https://raw.githubusercontent.com/alexeygrigorev/serverless-deep-learning/master/tflite/tflite_runtime-2.2.0-cp37-cp37m-linux_x86_64.whl --no-cache-dir

COPY clothing_model.tflite clothing_model.tflite
COPY lambda_function.py lambda_function.py

CMD [ "lambda_function.lambda_handler" ]
```

## Setup and Environment Requirements

just use Kaggle or Colab notebook and don't bother yourself with Huge Requirements.

## Usage

1. Clone the repository:
   ```
   git clone [Your Repository URL]
   cd [Repository Name]
   ```

2. Set up the environment using micromamba (adjust as needed):
   ```
   micromamba create -n fashion_class python=3.7
   micromamba activate fashion_class
   pip install -r requirements.txt
   ```

3. To train the model, run the `deep-learning.ipynb` notebook.

4. For serverless deployment, follow the steps in `serverless.ipynb`.

5. To test the model locally, use `test.py`.

6. To build and deploy the Docker image for Lambda:
   ```
   docker build -t fashion-classification .
   # Follow AWS ECR instructions to push the image
   ```

## API Usage

The model is deployed as an AWS Lambda function. To use it, send a POST request to the Lambda function URL with a JSON payload containing the 'url' key pointing to an image:

```json
{
  "url": "http://example.com/image-of-clothing.jpg"
}
```

The function will return a dictionary with class probabilities.


## Contact

[Your Name] - [Your Email]

Project Link: [Your GitHub Repository URL]
```