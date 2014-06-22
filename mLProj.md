A Predictive Model For Weight Lifting Dataset
========================================================
### Synopsis
Wearable accelerometers has the ability to measure human actvity. These afforable devices can provide a wealth of information about its users movements and activities. The purpose of this analysis is to build a predictive model based on a wearable accelerometers dataset where the users perform a simple weight lifting exercise either correctly (i.e. with correct form) or incorrectly. As shown in this report we can build a relatively simple and very accurate model to discriminate between correct, and the types of incorrect movements.
  
### Data Cleaning
The training dataset can be found [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) and the testing dataset can be found [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). More information about the dataset can be found [here](http://groupware.les.inf.puc-rio.br/har). This is the weight lifting excercises dataset. I add a few NA-string terms when reading in the dataset. This assumes the data file is in the working directory.


```r
pml<-read.csv("pml-training.csv",na.strings=c("NA","","#DIV/0!"))
dim(pml)
```

```
## [1] 19622   160
```
As evident we have a 19622 observations and 160 variables. Many of these variables are largerly NA.


```r
tmp1<-sapply(pml,FUN=function(x){sum(is.na(x))})
table(tmp1)
```

```
## tmp1
##     0 19216 19217 19218 19220 19221 19225 19226 19227 19248 19293 19294 
##    60    67     1     1     1     4     1     4     2     2     1     1 
## 19296 19299 19300 19301 19622 
##     2     1     4     2     6
```
We seem to have 60 out of the 160 variables with enough data points. These there are no columns with a few NA's that we could potentially impute I will remove all variable that contain any NA.

```r
pmlClean<-pml[,which(!tmp1>0)]
```
This is now our dataset with 60 variables. Inspecting the variables it is apparent that the there are variables we can omitt as they will not be useful when building this model.


```r
names(pmlClean)[1:7]
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

```r
pmlClean<-pmlClean[,-(1:7)]
```
We now have 52 variables all numeric and the response column "classe".

### Data Processing
  
We can now prepare the data for our model starting by dividing the data into a training and testing set.

```r
require(caret)
set.seed(1234)
inTrain<-createDataPartition(pmlClean$classe,p=0.7,list=F)
training<-pmlClean[inTrain,]
testing<-pmlClean[-inTrain,]
```
We can now do some preprocessing, namely center and scale the variables.

```r
preObj<-preProcess(training[,-53],method=c("center","scale"))
trainingSC<-predict(preObj,training[,-53])
testingSC<-predict(preObj,testing[,-53])
trainingSC$classe<-training$classe
testingSC$classe<-testing$classe
```
Note, that only the training set was used to calculate means and sd for the preObj. Further preProcessing such as PCA was explored but did not improve the model. Additionally the model computed rather fast.

### Training the Model
As this is a classification model with a realatively small dataset I am going to use a random forest model. I am not going to tune the model using the train() function of caret as this did not improve the model and was computationally costly, please se the appendix for more information about this. I will use the approximation of mtry as in sqrt(# of variables) which in this case is ~7.


```r
require(randomForest)
set.seed(5)
RFmtry7<-randomForest(trainingSC$classe~.,data=trainingSC,ntree=500,mtry=7)
RFmtry7
```

```
## 
## Call:
##  randomForest(formula = trainingSC$classe ~ ., data = trainingSC,      ntree = 500, mtry = 7) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.53%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    4    0    0    0    0.001024
## B   12 2641    5    0    0    0.006396
## C    0   15 2379    2    0    0.007095
## D    0    0   25 2226    1    0.011545
## E    0    0    3    6 2516    0.003564
```
Now, as running this algorithm is pretty fast, even on my 5 year old macbook, I tuned it manually, and found that mtry=6 gave me slightly better results. Also of note, increasing ntree does not improve the model.


```r
require(randomForest)
set.seed(3)
RFmtry6<-randomForest(trainingSC$classe~.,data=trainingSC,ntree=500,mtry=6)
RFmtry6
```

```
## 
## Call:
##  randomForest(formula = trainingSC$classe ~ ., data = trainingSC,      ntree = 500, mtry = 6) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 6
## 
##         OOB estimate of  error rate: 0.45%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    3    0    0    1    0.001024
## B   10 2645    3    0    0    0.004891
## C    0   10 2383    3    0    0.005426
## D    0    0   23 2228    1    0.010657
## E    0    0    2    6 2517    0.003168
```

### Cross Validating against Training Set
In order to get a better idea of how well the models perform on out of sample data we'll use the training set created above. Note that it has been centered and scaled based on the training set. We will predict the outcome using both models.

```r
pred7<-predict(RFmtry7,testingSC,type="response")
pred6<-predict(RFmtry6,testingSC,type="response")
result7<-table(testingSC$classe,pred7)
result6<-table(testingSC$classe,pred6)

get_missclass<-function(prediction,outcome){
        tmp<-table(prediction,outcome)
        sum(tmp[c(2:6,8:12,14:18,20:24)])/sum(tmp)
}

miss7<-get_missclass(pred7,testingSC$classe)
miss6<-get_missclass(pred6,testingSC$classe)
```
  
We can then see that the mtry=7 model gives us a missclassification rate of 0.0034 and the mtry=6 gives is a missclassification rate of 0.0029. Both are very small indicating that the out-of-sample error is small and there is not any overfitting.Looking at the confusion matrix:

```r
result6
```

```
##    pred6
##        A    B    C    D    E
##   A 1674    0    0    0    0
##   B    5 1133    1    0    0
##   C    0    5 1021    0    0
##   D    0    0    4  959    1
##   E    0    0    1    0 1081
```

```r
result7
```

```
##    pred7
##        A    B    C    D    E
##   A 1674    0    0    0    0
##   B    5 1133    1    0    0
##   C    0    6 1019    1    0
##   D    0    0    5  958    1
##   E    0    0    0    1 1081
```
we see that both models does extremely well in classifying the "correct"" movement (classe A). Based on these results I am satisfied with the prediction model and will do any further tuning or try other algorithms.

### Appendix: Tuning with caret
I made some attempts to train the model using the caret function train(). It did ok, but actually yielded a slightly worse model please find the code used to train the model here:


```r
require(caret)
ctrl<-trainControl(method="cv",
         number=10)
set.seed(10)
pmlFitC<-train(classe~.,
               data=trainingSC,
               method="rf",
               trControl=ctrl)
```
The outcome is ok but it does not perform as well as manually changing the mtry values. If you are grading this and have any suggestions how to better tune a model using caret, it would be greatly appreciated. Thanks.
