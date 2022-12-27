

[![LinkedIn][linkedin-shield]][https://www.linkedin.com/in/alex-chagan-a243221b6/]

<!-- PROJECT LOGO -->
<br />
<div align="center">
 
  <h3 align="center">Sports Image Classification</h3>
 
</div>



<!-- ABOUT THE PROJECT -->
## About The Project
An image classification project on 100 classes of sport genres. Uses a pretrained CNN EfficientNetB3 as base with fully connected layer for training. 
The dataset was taken from kaggle and it contains 13000 images with a split of 12000,500,500 for train, val, test.  
The model reaches 0.99 accuracy on test.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![TensorFlow][TensorFlow.org]][TensorFlow-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Download the kaggle dataset from: https://www.kaggle.com/datasets/gpiosenka/sports-classification/download?datasetVersionNumber=8 
 

### Installation


1. Clone the repo
   ```
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Install requirements
   ```
   pip install -r requirements
   ```
3. Place the train,val,test folders you downloaded from kaggle into sports-classifier-data folder 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

1. You can train your own model on the kaggle data or your own data with:
  ```
   python trainer.py
  ```

2. You can test your model on the test data with:
  ```
   python inference.py
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Alex Chagan  - alexchagan95@gmail.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/alexchagan/sports-images-classifier)

<p align="right">(<a href="#readme-top">back to top</a>)</p>




