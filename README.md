# BERTurk-Fine-Tune_Tweet-Classifier

## Description
### Project
A repository for fine tuning BERT models (BERTurk specifically) for multi-label text (tweet) classification. 

### Dataset
- The dataset must have the columns "text" and "label". 
- The "text" columns contains the tweets and the "label" column contains the labels for that tweet. 
- Both columns must hold string values. 
- If an entry has multiple labels, then the labels should be seperated with a comma (,). 
- A label cannot have blank spaces ( ), quotation marks ('' or "") and brackets ({} or []) in its name. 

|text|label|
|----|-----|
|Türk yüzyılında eğitim hak ettiği yeri alacak|eğitim|
|İstanbul'da çoğu metro durağında klimalar neden çalışmıyor?|ulaşım, çevre-ve-şehircilik|
|Türk milleti sığınmacılardan rahatsız ve ülkelerine gönderilmelerini istiyor. Bunun tek yolu seçim| mülteciler, seçim|


## Installation
### Dependencies
Python 3+ \
CUDA 11.6+   

### Setup
* Fill in the configuration parameters in .yaml files located at 'fine_tune_tweet_classifier/src/configs/' with the required information.
* __On Linux:__
  - Run "make install" if Python 3 is not installed.
  - Run "make init" to initialize a virtual environment with dependencies installed.
  - You may use "make run" in order to run the main file.

## Troubleshooting
### torch.cuda.OutOfMemoryError: CUDA out of memory
- If this exception is raised while forwarding the inputs into the model, try reducing the batch size.
- If it is raised while optimizing the weights (at optimizer.step() function call), try using a different optimizer (e.g. try using Stochastic Gradient Descend instead of ADAM).