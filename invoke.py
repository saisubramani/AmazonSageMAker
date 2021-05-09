import boto3
import json
import base64
from PIL import Image
import io
import cv2
import numpy as np


runtime_client = boto3.client('runtime.sagemaker')

img = '1.jpg'
with open(img,'rb') as f:
	payload = f.read()
	payload =bytearray(payload)

response = runtime_client.invoke_endpoint(EndpointName ='pytorch-inference-2020-05-14-11-24-27-060',ContentType = 'application/x-image',Body = payload)




str_response = response['Body'].read().decode()
print(str_response)


