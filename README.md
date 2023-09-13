Developing an AI model using YOLOv8 to detect humans in aerial drone images for Industrial Surveillance and potential Rescue Operations applications.

**Abstract**

Unmanned aerial vehicles (drones) equipped with high-resolution cameras and powerful GPUs 
have the potential to enhance Search-and-Rescue (SAR) operations in remote and hostile 
environments. Rapidly locating unconscious or injured victims is crucial for improving their 
chances of survival. This paper focuses on using drones as flying computer vision systems to 
increase detection rates and reduce rescue time. We conducted an experimental evaluation 
using the YOLOv5,Yolov7,Yolov8 algorithms, a lightweight version of the YOLO detection 
algorithm, on two newly created benchmark datasets specifically designed for SAR with 
drones. The results demonstrate promising outcomes, showcasing competitive detection 
accuracy comparable to state-of-the-art approaches but with significantly faster detection 
times.

**Problem Statement**

We have been given problem statement of Identifying and detecting humans in aerial view 
images captured by drones and UAV’s. Aerial images particularly of terrains , grass fields 
etc.So we need to develop model such it should be able to identify and detect humans.
In rescue and surveillance operations, the ability to detect and locate humans accurately and 
efficiently in large-scale environments is of paramount importance. Traditional ground-based 
approaches often face limitations in terms of coverage, accessibility, and real-time response. 
Therefore, there is a growing need for robust and effective human detection systems that 
leverage aerial images obtained from platforms such as drones or satellites.
However, detecting humans in aerial images poses unique challenges. Aerial images exhibit 
variations in scale, perspective, occlusion, and lighting conditions, making it difficult to apply 
conventional computer vision techniques designed for ground-level images. Furthermore, the 
presence of cluttered backgrounds, diverse poses, and the small size of humans in the images 
further exacerbate the complexity of the task.

**Objective**

The objective of this project is to develop a robust and efficient system for human detection 
using aerial image datasets, specifically focusing on aerial images obtained from platforms 
such as drones or satellites. The primary goal is to enable the detection of humans in real-time 
or near real-time, with a high level of accuracy and reliability. This system will be specifically 
designed for applications in rescue and surveillance operations, where the identification and 
tracking of humans in large-scale environments are crucial for effective decision-making and 
resource allocation.
So, our main objective is to develop a model to used in drone systems such that it should detect 
the human’s movements, actions. We use this technology in Industrial surveillances , to 
identify any movements of humans or animals over the Industrial Space.

**Methods and Methodology**

We followed implementation using 3 models :

1.YOLOv5 using Kaggle notebook

2.YOLOv7 using Kaggle notebook

3.YOLOv8 using Roboflow

1.YOLOv5

We have gone through different models and algorithms for detection technique and after going 
through few research papers we have decided to YoloV5 model.
YOLO is a regression-based method and is in fact much faster than region proposal-based 
methods (such as R-CNN ), although it is not as accurate. The idea behind YOLO is to realize 
object detection by considering it as a regression and classification problem: the first is used to 
find the bounding box coordinates for the objects in the image, while the second is used to 
classify the objects found in an object class. This is done in a single step by first dividing the 
input image into a grid of cells, and then calculating, for each cell, the bounding box and the 
relative confidence score for the object in that cell. Although the latest stable version of YOLO 
is YOLOv4 , we used YOLOv5 , which is still in development. The latest version of YOLO 
was chosen because several empirical results showed how it can work accurately, compared to 
YOLOv4, but with an extremely smaller model size.1 Many controversies about YOLOv5 have 
been raised by the community.2 These controversies are mainly caused by the fact that 
YOLOv5 does not even (yet) have a published paper; nevertheless, we have preferred to use 
the latter version to obtain experimental results that could be a valid reference for future work,
as other projects, such as, have done. In some experiments on the well-known COCO dataset , 
YOLOv5s showed much lower training time and model storage size than the custom YOLOv4 
model. Also, YOLOv5 takes less inference time, making it faster than YOLOv4.


YOLOv5 is different from all other previous versions; in particular, the major improvements 
introduced include mosaic data augmentation and the ability to autonomously learn bounding 
box anchors, i.e. the set of predefined bounding boxes of a certain height and width used to 
detect objects.

So we used yolov5 to train the model with heridal dataset.
Steps followed:
1.Prepairing Dataset
2.Installing and cloning Yolov5 from Ultralytics GitHub repository 
3. Train the YOLOv5 model
4. Test the YOLOv5 model
5. Evaluate the YOLOv5 model


Initially we trained yolov5 to coco2018 dataset by setting up required libraries pytorch etc 
Later trained on Heridal dataset. For this we generated yaml file which is used to train yolov5 
,yaml describes about our data train , test images and classes it has to yolo architecture.
Later we shifted our model yolov7 , due to some inconveniences and results of yolov7 
considered we shifted yolov7 for better performance.


2.YOLOv7

YOLOv7 is a deep learning-based object detection algorithm that can be used to detect humans 
in aerial view images.It is a single-stage object detector, which means that it can detect objects 
in a single pass through the image.YOLOv7 has been shown to be effective at detecting humans 
in aerial view images, even when the images are of low quality or when the humans are small 
or obscured.

YOLOv7 first divides the image into a grid of cells.For each cell, YOLOv7 predicts the 
probability of a human being present in the cell, as well as the coordinates of the bounding box 
around the human.YOLOv7 then uses a non-maximum suppression algorithm to remove 
overlapping bounding boxes.

Steps followed:
1. Preparing the dataset:
• Split your dataset into training and testing sets. Ensure that you have annotated
bounding boxes around the human objects in your training images.

• Organize the dataset according to the required YOLOv7 format. Each image 
should have a corresponding text file with the same name containing the 
annotations for the objects present in the image. The annotation format should 
include the class label (e.g., "human") and the coordinates of the bounding box 
(normalized values between 0 and 1).

• Additionally, you may need to generate patches from your images and annotate 
them as well if you plan to use patch-based training.

3. Setting up the YOLOv7 environment:
• Install the necessary dependencies, such as Python, PyTorch, and CUDA, if 
required. You can find the detailed installation instructions in the official 
YOLOv7 repository or documentation.

5. Obtaining the YOLOv7 source code:
• Clone the YOLOv7 repository from the official source. You can find the 
repository on GitHub or other platforms where it is available.

7. Configuring the YOLOv7 model:

• Customize the YOLOv7 configuration file according to your requirements. You 
may need to modify parameters such as the number of classes (in your case, 1 
for humans), the anchor sizes, and other hyperparameters. The configuration file 
is typically a .cfg file provided in the YOLOv7 repository.

8. Training the YOLOv7 model:
• Use the training script provided in the YOLOv7 repository to train your model. 
This script usually requires you to provide the path to your training and testing 
datasets, the path to the configuration file, and other relevant parameters.
• Start the training process and allow the model to learn from your annotated data. 
This may take some time depending on the size of your dataset and the 
complexity of the task.

10. Evaluating the trained model:
• Once the training is complete, you can evaluate the performance of your trained 
YOLOv7 model using the testing dataset.
• The evaluation metrics typically include precision, recall, and mean average 
precision (mAP). You can assess how well the model detects human objects in 
your aerial images.

12. Fine-tuning and optimization (optional):
• If the performance of the trained model is not satisfactory, you can try finetuning the model by adjusting the hyperparameters, augmenting the training 
data, or using other techniques to improve the detection accuracy.

14. Inference using the trained model:
• Once you are satisfied with the performance of the trained model, you can use 
it for inference on new, unseen aerial images to detect human objects.

• You can apply the trained model to full images or use the patches you generated 
during training to detect humans in a patch-based manner.

3.YOLOv8

Ultralytics YOLOv8 is the latest version of the YOLO (You Only Look Once) object detection 
and image segmentation model developed by Ultralytics. The YOLOv8 model is designed to 
be fast, accurate, and easy to use, making it an excellent choice for a wide range of object 
detection and image segmentation tasks. It can be trained on large datasets and is capable of 
running on a variety of hardware platforms, from CPUs to GPUs.


We are using roboflow which provides notebook along with models which we need to run.
Here are the detailed steps for each step:

1. Create a project in Roboflow.
To create a project in Roboflow, you will need to create an account and then click on the "Create 
Project" button. You will then be prompted to give your project a name and to select a 
workspace.

3. Upload your images to the project.
Once you have created a project, you can upload your images to the project. You can upload 
images in a variety of formats, including JPG, PNG, and TIFF.

5. Label your images using Roboflow's annotation tools.
 
Once you have uploaded your images, you will need to label them. Roboflow provides a variety 
of annotation tools that you can use to label your images. You can label objects by drawing 
bounding boxes around them, or you can label objects by assigning them tags.

7. Generate a new version of your dataset.
Once you have labeled your images, you will need to generate a new version of your dataset. 
This will create a new file that contains the images, the annotations, and the metadata for your 
dataset.

9. Export your dataset for use with YOLOv8.
Once you have generated a new version of your dataset, you can export it for use with YOLOv8. 
You can export your dataset in a variety of formats, including YOLOv4, YOLOv5, and 
YOLOv8.

11. Train YOLOv8 on your dataset.
Once you have exported your dataset, you can train YOLOv8 on your dataset. You can train 
YOLOv8 using a variety of tools, including the YOLOv8 command line tool and the YOLOv8 
training GUI.

13. Validate your model with a new dataset.
Once you have trained YOLOv8 on your dataset, you will need to validate your model with a 
new dataset. This will help you to ensure that your model is working correctly and that it is 
able to generalize to new data.

15. Deploy your model.

Once you have validated your model, you can deploy it. You can deploy your model in a variety 
of ways, including using a web service or using a mobile app.

