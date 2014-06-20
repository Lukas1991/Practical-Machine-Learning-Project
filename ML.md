Building Predictive Model For Weight Lifting Style using Accelerometer Data
========================================================
## Abstract and introduction
In this report, I will use machine learning algorithms to determine whether a particular form of exercise is performed correctly by using the accelerometer data.the data is originally from http://groupware.les.inf.puc-rio.br/har.

## Data preparation
1.The dataset can be downloaded as follows:

```r
if (!file.exists("./pml-training.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
        destfile = "./pml-training.csv")
}
if (!file.exists("./pml-testing.csv")) {
    download.file("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
        destfile = "./pml-testing.csv")
}
```


2.Load both datasets.

```r
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```


## Exploratory Analysis
1.The training set consists of 19622 observations of 160 variables, one of which is the dependent variable as far as this study is concerned:

```r
dim(training)
```

```
## [1] 19622   160
```


2.Partition training data provided into two sets. One for training and one for cross validation.

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(2534)
trainingIndex <- createDataPartition(training$classe, list = FALSE, p = 0.8)
new_training = training[trainingIndex, ]
new_testing = training[-trainingIndex, ]
```


3.Discarding columns with inadequate data

Most of the columns in the training data set are either empty or filled with invalid data (NA, empty space, etc.). We therefore clean the data by first observing which columns have most of their contents filled in, and only keeping those. 

```r
pmlColumnFillCounts <- sapply(new_training, function(col) {
    sum(!(is.na(col) | col == ""))
})
pmlFullColumns <- names(pmlColumnFillCounts[pmlColumnFillCounts == length(new_training$classe)])
```


4.Discarding columns to prevent overfitting

Additionally, it's worth noting that some of the variables in the data set do not come from accelerometer measurements and record experimental setup or participants' data. So I also discarded the following variables: X, user_name, raw_timestamp_part1, raw_timestamp_part2, cvtd_timestamp, new_window and num_window.

```r
pmlFullColumns <- pmlFullColumns[8:60]
```


5.Then I've cleaned up the data set, it would make sense to explore associations in the data.

```r
temp <- new_training[, pmlFullColumns]
temp <- temp[, 1:51]
pred.corr <- cor(temp)
pred.corr[(pred.corr < -0.95 | pred.corr > 0.95) & pred.corr != 1]
```

```
## [1]  0.9811 -0.9920 -0.9656  0.9811 -0.9754 -0.9656 -0.9920 -0.9754
```

There are eight variable pairs the Pearson correlation coefficient for which is above an arbitrary cutoff of 0.9 (in absolute value). 

```r
which(pred.corr > 0.95 & pred.corr != 1)
```

```
## [1]   4 154
```

```r
pred.corr[which(pred.corr > 0.95 & pred.corr != 1)]
```

```
## [1] 0.9811 0.9811
```

```r
which(pred.corr < -0.95)
```

```
## [1]  10  59 163 359 460 463
```

```r
pred.corr[which(pred.corr < -0.95)]
```

```
## [1] -0.9920 -0.9656 -0.9754 -0.9656 -0.9920 -0.9754
```

Interestingly, the roll_belt predictor participates in both of these pairwise interactions:

```r
pred.corr["roll_belt", "total_accel_belt"]
```

```
## [1] 0.9811
```

```r
pred.corr["roll_belt", "accel_belt_z"]
```

```
## [1] -0.992
```

From the analysis above, it seems good to discard at least the roll_belt variable to prevent excessive bias in the model.

```r
pmlFullColumns <- pmlFullColumns[2:53]
new_training <- new_training[, pmlFullColumns]
new_testing <- new_testing[, pmlFullColumns]
```

we should be aware that this analysis only explores pairwise, linear associations between variables. More general interactions is not computationally feasible.

## Predictive Model
We can build a random forest model using the numerical variables provided. As we will see later this provides good enough accuracy to predict the twenty test cases. And also we can obtain the optimal mtry parameter of 32 by using caret.

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed(25340)
```

Let's train a classifier using all of our independent variables and 2048 trees.

```r
rf_model <- randomForest(classe ~ ., data = new_training, ntree = 500, mtry = 32)
rf_model
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = new_training, ntree = 500,      mtry = 32) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 32
## 
##         OOB estimate of  error rate: 0.67%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4459    4    0    0    1    0.001120
## B   24 3004    7    1    2    0.011192
## C    0   13 2712   13    0    0.009496
## D    0    0   31 2541    1    0.012437
## E    0    0    4    4 2878    0.002772
```


## Cross Validation
It is able to measure the accuracy using our training set and our cross validation set. With the training set we can detect if our model has bias due to ridgity of our mode. With the cross validation set, we are able to determine if we have variance due to overfitting.

### In-sample accuracy

```r
training_pred <- predict(rf_model, new_training)
print(confusionMatrix(training_pred, new_training$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4464    0    0    0    0
##          B    0 3038    0    0    0
##          C    0    0 2738    0    0
##          D    0    0    0 2573    0
##          E    0    0    0    0 2886
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

The in sample accuracy is 100% which indicates, the model does not suffer from bias.

### Out of Sample accuracy

```r
testing_pred <- predict(rf_model, new_testing)
print(confusionMatrix(testing_pred, new_testing$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1115    8    0    0    0
##          B    0  748    3    0    0
##          C    0    3  679    9    2
##          D    0    0    2  633    3
##          E    1    0    0    1  716
## 
## Overall Statistics
##                                         
##                Accuracy : 0.992         
##                  95% CI : (0.989, 0.994)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.99          
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.986    0.993    0.984    0.993
## Specificity             0.997    0.999    0.996    0.998    0.999
## Pos Pred Value          0.993    0.996    0.980    0.992    0.997
## Neg Pred Value          1.000    0.997    0.998    0.997    0.998
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.173    0.161    0.183
## Detection Prevalence    0.286    0.191    0.177    0.163    0.183
## Balanced Accuracy       0.998    0.992    0.994    0.991    0.996
```

The cross validation accuracy is greater than 99%, which should be sufficient for predicting the twenty test observations. 
We should notice that the new data must be collected and preprocessed in a manner consistent with the training data.

## Test Set Prediction Results
Applying this model to the test data provided yields 100% classification accuracy on the twenty test observations.

```r
pmlFullColumns <- pmlFullColumns[1:51]
testing20 <- testing[, pmlFullColumns]
answers <- predict(rf_model, testing20)
answers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```


## Conclusion
We are able to provide very good prediction of weight lifting style as measured with accelerometers.It is well done.

by kannhaku



























