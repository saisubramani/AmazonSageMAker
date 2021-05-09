import logging
from PIL import Image
import cv2
import json
import pickle
import torch
import numpy as np
import os
import os.path
import io
import sys
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from wrn import WideResNet
from display_results import show_performance, get_measures, print_measures, print_measures_with_std
import lsun_loader as lsun_loader
import torch.utils.data as data





logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.info("inference.py script started")

JPEG_CONTENT_TYPE = 'application/x-image'
JSON_CONTENT_TYPE = 'application/json'


def model_fn(model_dir):
    logger.info("Entering into model_fn")
    #print("Entering into model_fn")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    #print("loading model")
    
    model = WideResNet(40,41,2,0.3)
    
    logger.info("1-model_dir path: "+str(model_dir))
    logger.info(os.system('pwd'))
    logger.info(os.system('ls'))

    pa = os.path.join(model_dir,'model_1.pt')
    logger.info("path:"+str(pa))
    #print("path:"+str(pa))
    with open(pa,'rb') as f:
        model.load_state_dict(torch.load(f))
        model.eval()

    logger.info('Done loading model')
    #print('Done loading model')
    return model


def input_fn(request_body,request_content_type=JPEG_CONTENT_TYPE):
    logger.info("Entering into input_fn")
    if request_content_type == 'application/x-image':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #print(type(request_body))
        img_arr = np.array(Image.open(io.BytesIO(request_body)))
        BGR = cv2.cvtColor(img_arr,cv2.COLOR_RGB2BGR)
        img_resize = cv2.resize(BGR,(32,32))
        #cv2.imwrite("resized.jpg",img_resize)
        img_transpose = np.transpose(img_resize, (2, 0, 1))
        img_tensor = torch.tensor(img_transpose).float()
        processed_img = (img_tensor, 0)

        return processed_img

    raise Exception('Requested unsupported contentType: '+request_content_type)


def predict_fn(input_data,model):
    logger.info("Entering into Preiction_fn")
    #print("Entering into Preiction_fn",model)
    input_data_out = []
    data_out = []
    result = []
    model = model
    data_out.append(input_data)
    input_data_out = data_out[0: int(len(data_out) * 1.0)]
    train_loader_out = torch.utils.data.DataLoader(input_data_out, batch_size=200, shuffle=True,num_workers=4, pin_memory=True)
    cudnn.benchmark = True  
    ood_num_examples = len(train_loader_out) // 5
    expected_ap = ood_num_examples / (ood_num_examples + len(train_loader_out))
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()
    

    logger.info("prediction started")
    loader = train_loader_out
    in_dist=True
    _score = []
    out_conf_score = []
    in_conf_score = []
    _right_score = []
    _wrong_score = []
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            try:
                output = model(data)
                logger.info("data passed to model")
            except Exception as e:
                logger.info("error:",str(e))
            smax = to_np(F.softmax(output, dim=1))
            if '':
                _score.append(
                    to_np((output.mean(1) - torch.logsumexp(output, dim=1))))
            else:
                _score.append(-np.max(smax, axis=1))
                out_conf_score.append(np.max(smax, axis=1))
            if in_dist:
                in_conf_score.append(np.max(smax, axis=1))
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)
                if '':
                    _right_score.append(
                        to_np((output.mean(1) -
                               torch.logsumexp(output, dim=1)))[right_indices])
                    _wrong_score.append(
                        to_np((output.mean(1) -
                               torch.logsumexp(output, dim=1)))[wrong_indices])
                else:
                    _right_score.append(-np.max(smax[right_indices], axis=1))
                    _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
    if in_dist:
        print(1)
        in_conf_score = concat(in_conf_score).copy()
        in_score = concat(_score).copy() 
        right_score = concat(_right_score).copy()
        wrong_score =concat(_wrong_score).copy()
    else:
        print(2)
        in_conf_score, in_score, right_score, wrong_score = concat(out_conf_score).copy(), concat(_score)[:ood_num_examples].copy()


    score = list(in_conf_score)
    score_str =str(score[0])
    
    for i in in_conf_score:
        if i < 0.2:
            outlier_status = "outlier"
        else:
            outlier_status = "not_outlier"


   #print("result:",outlier_status)
    logger.info("result:",outlier_status)
    result.append(outlier_status)
    result.append(score_str)
    return result

def output_fn(prediction,content_type=JSON_CONTENT_TYPE):
    #print("Entering into output_fn")
    logger.info("Entering into output_fn")
    if content_type == JSON_CONTENT_TYPE:
        return json.dumps({'results':prediction})

    raise Exception('unsupported contentType: '+content_type)


