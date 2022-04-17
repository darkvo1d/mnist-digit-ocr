# OCR

This is a web deployed project which allows user to draw a math digit ranging from 0-9 on a HTML canvas. The user drawn image is then processed and passed through an ML model to predict the user drawn digit.

## Installation

Download repo and run the following command to install dependencies.

```bash
pip install requirements.txt
```

The below command trains the model and saves an *.h5* file
```python
python model.py
```

## Usage

The working demo of the project can be accessed here : http://mnist-digit-ocr.canadacentral.azurecontainer.io

## Project details

The data on which the model is trained is the MNIST dataset for math digits.
The model used is a Sequential model with 2 2D conv layers and 2 Dense layers.

## License
[MIT](https://choosealicense.com/licenses/mit/)
