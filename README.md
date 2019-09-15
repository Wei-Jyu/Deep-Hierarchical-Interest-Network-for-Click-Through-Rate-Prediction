# Deep Hierarchical Interest Network for Click-Through Rate Prediction
Implementation of Deep Hierarchical Interest Network for Click-Through Rate Prediction using tensorflow.
## Data
- [Amazon Data](http://jmcauley.ucsd.edu/data/amazon/)<br/>
### Prepare Amazon Data
- You can execute the following command to get Amazon Book data prepared<br/>
```
sh prepare_amazon_book.sh
```
## Running Code
Execute the command below to train/test models:
```
train.py [-h] [-task TRAIN|TEST] [--model_type MODEL_TYPE] 
```
For example, if you want to train model DHIN, use the following command:
```
python script/train.py -task train --model_type DHIN
```
The model below had been supported: 
- DNN 
- PNN 
- DIN
- DIEN
- DIEN_with_InnerAtt
- DHIN_without_InnerAtt
- DHIN

Note: we use python 2.x and tensorflow 1.4.
