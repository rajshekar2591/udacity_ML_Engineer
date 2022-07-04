# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

I used resnet18 pretrained model initialized with imagenet weights and used transfer learning on the dog image dataset modifying the weights of the fully connected layers. I used this model because it's pretrained on ImageNet which is a large dataset and thus can identify general patterns on the dog Image dataset pretty well with transfer learning. 

Used below hyperparameters along with the range:

Learning Rate: linear search space from 0.001 to 0.1

batch-size: categorical search space of 16,32,64

epochs: Integer search space of 2,4

Remember that your README should:
- Include a screenshot of completed training jobs
- Logs metrics during the training process
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

![alt text](https://github.com/rajshekar2591/udacity_ML_Engineer/blob/main/dog_breed_classification/images/Screen%20Shot%202022-07-03%20at%201.47.19%20PM.png)

![alt text](https://github.com/rajshekar2591/udacity_ML_Engineer/blob/main/dog_breed_classification/images/Screen%20Shot%202022-07-03%20at%201.48.39%20PM.png)

![alt text](https://github.com/rajshekar2591/udacity_ML_Engineer/blob/main/dog_breed_classification/images/Screen%20Shot%202022-07-03%20at%201.49.01%20PM.png)

![alt text](https://github.com/rajshekar2591/udacity_ML_Engineer/blob/main/dog_breed_classification/images/Screen%20Shot%202022-07-03%20at%201.49.19%20PM.png)

![alt text](https://github.com/rajshekar2591/udacity_ML_Engineer/blob/main/dog_breed_classification/images/Screen%20Shot%202022-07-03%20at%201.49.35%20PM.png)



## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

Performed debugging and profiling in Sagemaker using Sagemaker Debugger and Profiler respectively. 

To perform debugging using Sagemaker Debugger, used the following steps.

1. Added hooks for the debugger and the profiler in the train() and test() functions and set them to their respective modes. 

2. In the main() function created the hook and registered the model to the hook. This hook is passed to the train() and test() functions.

3. In the notebook, configured the debugger rules and the hook parameters.

For the profiler, used the following steps.

1. Created profiler rules and config.

2. The debugger configuration, the profiler configuration and the rules passed to the estimator.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

The profiler report shows the time it took for completing the training and also the hardware utilization while training. The training loss was decreasing and so as the validation loss apart from a minor peak and then going down post that.

![alt text](https://github.com/rajshekar2591/udacity_ML_Engineer/blob/main/dog_breed_classification/images/Screen%20Shot%202022-07-03%20at%202.09.22%20PM.png)


The profiler report can be found at dog_breed_classification/ProfilerReport/profiler-output/profiler-report.html



**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

The model was deployed to an endpoint on Sagemaker with an instance type of "ml.m5.large". The below image shows the deployed endpoint in Sagemaker. The notebook train_and_deploy contains code about how to call the endpoint inferencing an example image.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

![alt text](https://github.com/rajshekar2591/udacity_ML_Engineer/blob/main/dog_breed_classification/images/Screen%20Shot%202022-07-03%20at%201.51.12%20PM.png)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
