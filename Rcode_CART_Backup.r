################### Info ################################
# R code for MACHINE LEARNING 
# by Marica Valente

# Based on own code, as well as code and data set from:
# "Big Data: New Tricks for Econometrics
# Journal of Economic Perspectives 28(2), 3-28            
# http://pubs.aeaweb.org/doi/pdfplus/10.1257/jep.28.2.3
# Hal R. Varian

# For ML in R, check this out:
# https://bradleyboehmke.github.io/HOML/
# "Hands-On Machine Learning with R"

################## Description #################################

################### The Titanic #######################

# In the night of 14 April through to the morning of 15 April 1912, the RMS Titanic sank on its
# maiden voyage from Southampton to New York. Of the estimated 2,224 people on board, more than
# 1,500 found death in the North Atlantic Ocean. This Problem Set aims on identifying the variables
# that best predict survival and death of the passengers using tree-based methods.
# Your task is to train algorithms to predict survival / death of passengers as accurately as possible.
# We provide a dataset of approx. 1200 passengers of the Titanic that contains for each passenger a variety
# of information, including, e.g., a passenger's age, name, and family status, the ticket price that
# was paid, and a binary variable that equals 1 if the passenger survived. 

# Deriving further features/predictors from the given data may increase predictive power of the algorithm
# To check the accuracy of the trained algorithm, another dataset of around 190 passengers is used
# as ultimate test data. This dataset contains the same information as the main dataset - except
# for the survival dummy variable. Send your predictions to me or in OLAT!


################ Packages + data #################################
rm(list = ls())
getwd()
#setwd("../seadrive_root/Marica V/My Libraries/Teaching_WS22/Potsdam")
setwd("/Volumes/Extreme SSD/UIBK/Semester von PC/UBIK/Semester 4/VU Aktuelle Entwicklungen in Wirtschaft und Gesellschaft Maschinelles Lernen für Progn/R/data")

# Install these packages from Tools-->Install Packages..
# Then load them using library()
library(rpart) # tree functions
library(rpart.plot) # tree plot functions
library(randomForest) # RF functions
library(caret) # RF: to get confusion matrices
library(pdp) # RF: to derive partial dependence plots
library(xtable) # graphic package to build tables
library(knitr) # graphic package to better display R objects
library(splines)
library(nnet) # logit regression
library(ROCR) # for logit
library(e1071)
library(devtools) # to install packages from github

# Install edarf directly from github
# Edarf may be a further tool to analyze RF (Random Forests)
# Then load it using library()
install_github("zmjones/edarf", subdir = "pkg")
library(edarf)

# (optional) 
# Replication exercise: http://cran.nexr.com/web/packages/edarf/vignettes/edarf.html

################# Load datasets
#### Exercise dataset

# Training data (to further split into train set for IS prediction and validation set for OOS prediction)
trdata<-read.csv2("titanic_training_data_1000.csv", header = TRUE, sep=";",dec=".")

#### Assignment dataset
# Test data for prediction (not used for training or tuning)
# This is the test data for your assignment
# Outcome vector is removed for assignment
tedata<-read.csv2("titanic_test_data_189.csv", header = TRUE, sep=";",dec=",")

# Check variable definition in PDF Titanic_Data_Description
#### Anpassung Cabin ####
trdata$cabin <- substr(trdata$cabin, 1, 1)
# Erstellen Sie eine Funktion zur Zuordnung der Decks basierend auf der Kabinennummer
assignDeck <- function(cabin) {
  # Falls die Kabinennummer nicht fehlt (NA)
  if (!is.na(cabin)) {
    # Extrahieren Sie den ersten Buchstaben der Kabinennummer
    deck <- substr(cabin, 1, 1)
    
    # Führen Sie die Zuordnung basierend auf den gegebenen Informationen durch
    if (deck %in% c("A")) {
      return("Oberdeck")
    } else if (deck %in% c("B")) {
      return("Oberdeck")
    } else if (deck %in% c("C")) {
      return("Oberdeck")
    } else if (deck %in% c("D")) {
      return("Oberdeck")
    } else if (deck %in% c("E")) {
      return("Mittleredeck")
    }else if (deck %in% c("F")) {
      return("Mittleredeck")
    }else if (deck %in% c("G")) {
      return("Mittleredeck")
    }else if (deck %in% c("T")) {
      return("Unterdeck")
    }else if (deck %in% c("U")) {
      return("Unterdeck")
  } else {
    return("Unbekanntes Deck")
  }
    }
}

# Verwenden Sie die Funktion, um ein neues Dataframe mit der Zuordnung zu erstellen
trdata$deck_assignment <- sapply(trdata$cabin, assignDeck)

# Überprüfen Sie das neue Dataframe
head(trdata)


#####Anpassung Title #####

# Funktion zum Extrahieren des Titels und Aktualisieren des Namens
# Funktion zum Extrahieren des Titels
extractTitle <- function(name) {
  # Erstellen einer Regel, um den Titel zu extrahieren
  title <- sub(".*, ([A-Za-z]+)\\..*", "\\1", name)
  
  return(title)
}

# Anwendung der Funktion auf Ihr Datenframe, um eine Spalte für die Titel hinzuzufügen
trdata$title <- sapply(trdata$name, extractTitle)

# Überprüfen Sie die Titel in Ihrem Datenframe
head(trdata$title)

#####Anpassung Family Size #####
trdata$family_size <- trdata$sibsp + trdata$parch + 1
trdata$last_name <- sapply(strsplit(trdata$name, ','), function(x) trimws(x[1]))



################ Preliminaries
# Test data should be completely different than training data
tedata$name %in% trdata$name # ALL FALSE, passengers differ
head(trdata)
str(trdata) # tells the class of each variable
colnames(trdata)
colnames(tedata)

dim(trdata) # 1000 rows
dim(tedata) # does NOT contain outcome (i.e. variable "survived")


#### 0. Data analysis ####

1-length(trdata$survived[trdata$survived==0])/1000 
# share survived

prop.table(table(trdata$survived)) 
# without splits, a tree would predict?

prop.table(table(trdata$survived[trdata$pclass>2])) 
# share survived in 3rd class

prop.table(table(trdata$survived[trdata$age<2])) 
# share babies survived

prop.table(table(trdata$survived[trdata$age>18])) 
# share adults survived

prop.table(table(trdata$sex)) 
# gender composition on boat

prop.table(table(trdata$sex, trdata$survived), 1) 
# most males died and females survived


# Divide training and validation set (30% validation set and 70% training set)
set.seed(100) # allows replication of results on any pc
# put whatever nr, but never change this number once you selected one
tr <- sample(x=1:nrow(trdata), size=0.70*nrow(trdata), replace=F)
# 70% of the data are IS, used for training

te <- (1:nrow(trdata))[-tr]
# 30% of the data are OOS, used for prediction
length(tr)
length(te)

# Randomization worked OK
prop.table(table(trdata$survived))
prop.table(table(trdata[tr,]$survived))
prop.table(table(trdata[te,]$survived))
# very similar to the original data (standard ass: tr and test data come from same distribution)


#### 1. Build UNpruned CT ####
?rpart # check out the RPART vignette https://cran.r-project.org/web/packages/rpart/rpart.pdf
?rpart.control

# For more details on RPART, check this out:
# https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf

colnames(trdata) #  Use the most straightforward predictors
str(trdata)
# No pruning, i.e. cp=0 (and no cross-val. xval=0), full tree depth
CT_no_CV <- rpart(survived ~ pclass  + age + sex + fare + embarked + sibsp, 
                  data = trdata[tr,], 
                  method = "class" ,  # if Y is binary
                  model = FALSE,  #  keep a copy of the model frame in the results
                  x = TRUE,  # keep a copy of x matrix in the results
                  y = TRUE, # # keep a copy of y vector in the results
                  control=rpart.control(minsplit=1, #min. n. of obs. that must exist in a node in order for a split to be attempted 
                                        minbucket =1, #min. n. of obs. in leaf: for full tree, and MCE = 0 (no bias)
                                        cp= 0, #pruning parameter/complexity parameter (class. error must decrease by cp at each step or no further split is performed)
                                        xval = 0, #k-nr of fold CV for later pruning (here no pruning, full tree)
                                        maxdepth=30)) # max by default
# maxdepth = max nr of sequential splits performed                                        
# by default maxdepth=30 is the max allowed computationally
# as a result, some leafs can contain more than one observation (no perfect prediction)
# that's why leaves can still predict with some positive error

rpart.plot(CT_no_CV) # full tree
# normally the plot here is too big to be visible bec. tree has full depth before pruning

### rpart Output
names(CT_no_CV)
# Structure of the tree (splits performed)
# Shows order of splits, type of node (split or leaf), n. obs in each node, prediction at each node, ncompete found
CT_no_CV$frame[1:10,1:6]
# "var": names of the variables used in the split at each node (leaf nodes are denoted by the level "<leaf>")
# "n": n. obs. reaching the node
# "dev": deviance or RSS
# "yval": 1/2 (died/survived) the fitted value (prediction) of the response at the node
# "complexity" (alpha) = % reduction in prediction error with that split
# without CV and perfect prediction: dev=0 and alpha=0 in a leaf 
# For example: leaf 256 contains n=20 obs., MCE=0, yval=1=died, alpha=0

# Have a look at all leafs
CT_no_CV$frame[CT_no_CV$frame$var=="<leaf>",1:6]


# Look at the split by sex (first split)
# Complexity is 0.456 (45.6%)

# How to compute this by hand
# Example: full tree, MCE:
table(trdata[tr,]$survived)
285/(415+285)
# Rescale 0.407 to 1
# Splitting on "sex" reduced the MCE by 1-(error when splitting by sex)

table(trdata[tr,]$sex, trdata[tr,]$survived)
# MCE female (right node) + male (left node)
(70+85)/700 # 0.22

# New relative error
0.22/0.407 # 0.54

# Decrease in classif error ("complexity")
1-0.5405405 # 46%
# 1-(error when splitting by sex)


CT_no_CV$where   # In which leaf node does an observation fall
length(CT_no_CV$where) # length would be 1000 if all data are used for training
# otherwise only 700 because 300 are kept for prediction OOS

CT_no_CV$cptable # MCE for different complexity parameters (CP)
# CP ("complexity") is reduction in rel error thanks to that split
(1-CT_no_CV$cptable[2,3]) #reduction of MCE from split0 to split1

# nsplit is nr of splits performed after min of the training error
# Standardized: Max. error = 1
# For each alpha, there is unique tree size T_alpha (Breiman's result)
# For alpha=0, there are many nsplit = max tree depth
# LAST rel error corresponds to maxdepth=30 (thus, nonzero)

# Find min alpha? Always the full tree (we are in the training set!)
# Overfit, smallest MCE, all observations perfectly predicted, but no external validity


# Predict In-Sample (IS)
prediction_IS <- predict(CT_no_CV, newdata = trdata[tr,], type = "class")

# Is an individual correctly classified 0/1?
trdata[tr,]$survived == prediction_IS

# 1-accuracy is share misclassification = Misclassif. error (all training data used)
1- sum(trdata[tr,]$survived == prediction_IS)/700
sum(trdata[tr,]$survived == prediction_IS)/700
# Good accuracy in-sample (>0.50 algorithm does better than at random)

# Predict Out-of-sample (OOS)
prediction0 <- predict(CT_no_CV, newdata = trdata[te,], type = "class")

# OOS: 1-accuracy is share misclassification = Misclassif. error (all training data used)
1- sum(trdata[te,]$survived == prediction0)/700
# MCE OOS is lower than MCE IS
sum(trdata[te,]$survived == prediction0)/700
# Lower accuracy OOS (as expected)




#### --> Prediction contest ####
# Predict for other database
predictionbo0 <- predict(CT_no_CV, newdata = tedata, type = "class")
colnames(tedata)

# How many survived in the other database?
predictionbo0 # Compare this vector of predictions with the real one (not given yet)

# Predicted share of survivors
sum(predictionbo0 == 1)/length(predictionbo0)
# Compare this share to the real data!
# Real share is: 0.312



#### 2. Build Pruned CT ####

### Pruned/Cross-validated tree
CT_with_CV <- rpart(survived ~ pclass  + age + sex + fare + embarked + sibsp,  
                    data = trdata[tr,], 
                    method = "class" ,  # if Y is factor
                    model = FALSE, 
                    x = TRUE, 
                    y = TRUE, 
                    control=rpart.control(minsplit=2, #no need to specify minbucket
                                         # cp= 0,
                                          xval = 5))

# Remember: 5-fold CV means
# (i) for a given alpha, fit model in each of 4 folds
# (ii) get prediction accuracy using data of the 5th fold (validation set), 
# (iii) obtain average prediction accuracy by averaging the 4 prediction accuracy values
# (iv) compare averaged prediction accuracies for different values of alpha

CT_with_CV$cptable # with 5-fold CV (each fold contains 700/5=140 obs. - different obs. in each fold!)
# predict in 4 folds, evaluate prediction OOS in the remaining fold
# 5 folds, 5 evaluation rounds 
# 5 OOS predictions on the leftout fold that will be then averaged
# Repeat this for different values of CP (alpha) until CP=0.01 by default
# 1% is the default limit for deciding when to consider splits.

# nsplit is subject to the rule minsplit=2
# xerror is the cross-validated MCE we aim to min with the cost-complexity minimization
# xstd is the st dev of the xerrors across samples (rounds)

#The complexity table is printed from the smallest tree (no splits) to the largest one
#(8 splits).

# The number of splits is listed, rather than the number of nodes. The number of nodes
# is always 1 + the number of splits.


plotcp(CT_with_CV) # only for CV, alpha greater than 0
# yaxis plots xerror (CValidated) and xaxis plots alpha level
# plot shows that the min. xerror corresponds to a tree size of about 9 leafs (or 8 splits)


# Find min alpha after k-fold CV
# Among the k values of alpha pick the one associated to the lowest OOS (out-of-fold) xerror (classif. error)
minMSE<-min(CT_with_CV$cptable[,4]) # min xerror
minMSE
optcp<-CT_with_CV$cptable[which(CT_with_CV$cptable[,4]==minMSE),1]
optcp # alpha for min xerror
as.numeric(CT_with_CV$cptable[,2][CT_with_CV$cptable[,4]==minMSE]) 
# nr of tot splits performed for optcp chosen

# Now we prune the tree with this optimal complexity value
opt_tree <- prune(CT_with_CV, cp=optcp)
rpart.plot(opt_tree) 
# nr tot splits with opt.alpha (pruned) <<< nr tot splits with alpha=0 (unpruned)


rpart.plot(CT_no_CV)
# difference with full/unpruned tree



# IS prediction 
prediction_IS_pruned <- predict(opt_tree, newdata = trdata[tr,], type = "class")
sum(trdata[tr,]$survived == prediction_IS_pruned)/length(tr) 
# TR. SET ACCURACY is 0.85
# Lower accuracy IS than for unpruned tree (0.98)

# Predict with share of left-out training data (test data)
prediction1 <- predict(opt_tree, newdata = trdata[te,], type = "class")

# Check prediction accuracy OOS:
sum(trdata[te,]$survived == prediction1)/length(te) 
# TEST SET: ACCURACY is 0.796666
# better accuracy OOS than unpruned tree (0.31)



# Summary about optimal tree which includes var. importance measure
summary(opt_tree) # opt. tree = pruned tree with opt. alpha (cross-validated)


### Rank/variable importance:
# Variable importance
# sum over the reductions of misclassified individuals attributed to each variable at each split
varImp(opt_tree)
# sum of the increase in predictive accuracy when the predictor of interest is splitted upon
opt_tree$splits[, "improve"] 



#### --> Prediction contest ####
# Predict for other database
predictionbo1 <- predict(opt_tree, newdata = tedata, type = "class")

# How many survived in the other database?
sum(predictionbo1 == 1)/length(predictionbo1)
# Compare this to the real data (here not given)

# Compared to OOS prediction of UNpruned tree
sum(predictionbo0 == 1)/length(predictionbo0)




#### --> Analyze plotted optimal tree ####
rpart.plot(opt_tree)


### Plotted node shows three values:
# Top value: Prediction 0(dead) or 1(survived)
# Middle value: Share of 1s (survived) in the data
# Bottom value: Share of all obs. falling in that  (node probability)

### Middle value:
# Root node predicts death
sum(trdata[tr,]$survived)
round(285/700,2)
# Here MCE = share of survived
# Root tree: Missc. error (MCE) = survived(285)/tot obs(700) = 0.41

# Other ways to display the tree and the data contained in each node
rpart.plot(opt_tree,extra=1) # 415 ppl died, 285 survived: 345 males died, 85 survived; 70 females died, 200 survived
rpart.plot(opt_tree,extra=2) # 415 over 700 ppl died; 200 over 270 females survived, 345 over 430 males died


rpart.plot(opt_tree)
# Split=1 Look at the shares (middle value).
# This is computed from:
length(trdata[tr,]$survived[trdata[tr,]$sex=="female"])
# 270 tot. females in tr data
sum(trdata[tr,]$survived[trdata[tr,]$sex=="female"]) # n. survived females
# 200 surviving females in tr data
200/270 # 0.74
# what is shown here SHARE SURVIVORS
rpart.plot(opt_tree)

# 0.74 females and 0.2 males
# This is computed from:
length(trdata[tr,]$survived[trdata[tr,]$sex=="male"]) # tot. males in tr data
sum(trdata[tr,]$survived[trdata[tr,]$sex=="male"]) # n. survived males
85/430 # 0.2


### Bottom value:
rpart.plot(opt_tree)
# Root tree: 100% share of all obs. falling in that node
# Split=1: LEFT NODE (% of males) and RIGHT NODE (% females)
prop.table(table(trdata[tr,]$sex))


# Compute pred. accuracy gain after first split
# Make an hypothesis: If we would stop at root node we would predict that:
# all people die --> MCE=?
sum(trdata[tr,]$survived)/700 # 0.41

# Make an hypothesis: If we would stop after first split we would predict that:
# split=1: all F live, all M die --> MCE=?
# N. dead females (mispredicted if tree would stop here):
sum(1-trdata[tr,]$survived[trdata[tr,]$sex=="female"])

# N. survived males (mispredicted if tree would stop here):
sum(trdata[tr,]$survived[trdata[tr,]$sex=="male"]) 

# Overall MCE
(70+85)/700 # 0.22 is MCE

# We can obtain the same just by looking at the plot!
0.61*0.20+0.39*(1-0.74) # 0.22
#0.61 is prop of data in left node = male, 0.20 is prop of males misclassified (who live instead of dying)
#0.39 is prop of data in right node = female, (1-0.74) is prop of females missclassified (who die instead of living)

# MCE goes down from 0.41 (root node) to 0.22 (first split)
(0.41-0.22)

# Standardize the measure by dividing by MCE of Root tree:
round((0.41-0.22)/0.41,2)







################# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ####################################

#### 3. New covariates ####

#### 4. Build Pruned CT with new covariates ####

# Marica created some new covaraites in code
# Rcode_trees_newcovariates (provided later on as a feedback)

# Add your own covariates and try whether predictions
# of trees improve, even by a super small amount

################# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ####################################




#### 5. Build RF ####
library(randomForest)
?randomForest

class(trdata$survived) # for RF has to be factor
head(trdata$survived)
trdata$survived<-as.factor(trdata$survived)
trdata$pclass<-as.factor(trdata$pclass)
trdata$sex<-as.factor(trdata$sex)
trdata$cabin<-as.factor(trdata$cabin)
trdata$title<-as.factor(trdata$title)
trdata$deck_assignment<-as.factor(trdata$deck_assignment)
trdata$last_name<-as.factor(trdata$last_name)
trdata$family_size<-as.factor(trdata$family_size)



#trdata$parch<-as.factor(trdata$parch)
RF<-randomForest(survived ~ pclass + age + sex + embarked +  fare + sibsp + cabin + title + deck_assignment + family_size + boat,
                 data = trdata, 
                 ntree=1000,#1000 original #500 is standard, ntree is one of the 2 free parameters
                 # for bootstrapped sample, grow unpruned tree split acc. to only mtry predictors used at each node
                 # for obs. i get predictions with the trees in which i was OOB, majority rule/avg over these trees
                 mtry= 2, #2 original #number of variables in each split #second of the 2 free par. = sq.root(par) for classif or p/3 for regression
                 # variables randomly sampled as candidates at every split
                 # if variable very correlated choose less
                 #maxnodes=4,
                 proximity=T,  # The proximity in Random Forests is defined to be the proportion, taken over all the trees in the forest,
                 # of the times that two observations end up in the same terminal node.
                 oob.prox=T, # compute proximity on OOB data only
                 importance=TRUE,
                 replacement=F # Should sampling of cases be done with or without replacement?
)

print(RF) 

# The proximity in Random Forests is defined to be the proportion, taken over all the trees in the forest,
# of the times that two observations end up in the same terminal node.



# Every tree is built and includes different predictors, "starts" with different predictors
# Given test data, the path from the initial value of a given predictor with which a tree starts
# to the final prediction leads to different conclusion regarding survived
# If the majority of the trees of the RF predicts survived, then the RF draws this conclusion

print(RF) 


#### OOB Error ####

attributes(RF)
RF$confusion # conf. matrix tells correctly classified outcomes
print(RF) # OOB error = 19.5%

# Where does this OOB error come from?
dim(trdata)
RF$confusion # 1- OOB is accuracy , i.e. 1 - (the sum of diag.el divided by tot obs)
oob_accuracy<-(RF$confusion[1,1]+RF$confusion[2,2])/1000
1-oob_accuracy # OOB error

# Confusion matrix: Actual values on top
# false negative
RF$confusion[1,2]/(RF$confusion[1,2]+RF$confusion[2,2]) # 57 misclassified to be dead but they were alive
# false positive
RF$confusion[2,1]/(RF$confusion[2,1]+RF$confusion[1,1]) # 134 misclassified to be alive but they were dead

# --> highest MCE is the false positive: more mistakes on who should have been classified as dead
# Predicting death correctly is slightly more difficult

# sensitivity = given that person is dead in reality (reference column = 0), how good we predict it
549/(549+129) # sensitivity = no-false-positive (true positive rate)
129/(549+129) # false positive
# false positive = 1 - sensitivity

# specificity = given that person is alive in reality (ref. column =1), how good we predict it
264/(264+58) # specificity
58/(58+264) # false negative
# 1- false negative = specificity


# Predict based on OOB units
prediction_RF_train <- predict(RF,  type = "class")
length(prediction_RF_train) # OOB predictions for all training set

#TRAINING SET: MCE is 1-0.81
sum(trdata$survived == prediction_RF_train, na.rm = T)/1000

# install package caret: Get confusion matrix for prediction with ANY data you want
# install.packages("caret")
# IS confusion matrix vs. OOB error is, of course, computed with OOS observations
confusionMatrix(prediction_RF_train, trdata$survived)
?confusionMatrix
# What does this confusion matrix include?
# The overall accuracy rate is computed along with a 95 percent confidence interval for this rate 
# (using binom.test) and a one-sided test to see if the accuracy is better than 
# the "no information rate," which is taken to be the largest class percentage in the data.

# The overall accuracy and unweighted Kappa statistic are calculated. A p-value from 
# McNemar's test is also computed using mcnemar.test (which can produce NA values with sparse tables).
# Reference: Alan Agresti (1990). Categorical data analysis. New York: Wiley. Pages 350-354.

# Predicted	 Event	 No Event
# Event	        A	       B
# No Event	    C	       D

# The formulas used here are:
# Sensitivity = A/(A+C)
# 
# Specificity = D/(B+D)
# 
# Prevalence = (A+C)/(A+B+C+D)
# 
# PPV = (sensitivity * prevalence)/((sensitivity*prevalence) + ((1-specificity)*(1-prevalence)))
# 
# NPV = (specificity * (1-prevalence))/(((1-sensitivity)*prevalence) + ((specificity)*(1-prevalence)))
# 
# Detection Rate = A/(A+B+C+D)
# 
# Detection Prevalence = (A+B)/(A+B+C+D)
# 
# Balanced Accuracy = (sensitivity+specificity)/2
# 
# Precision = A/(A+B)
# 
# Recall = A/(A+C)
# 
# F1 = (1+beta^2)*precision*recall/((beta^2 * precision)+recall)
# 
# where beta = 1 for this function.


# Alternatively, compute accuracy by hand:
# Prediction compared to reality for training data
head(trdata$survived)
head(prediction_RF_train)
sum(trdata$survived == prediction_RF_train, na.rm = T)/1000
# 0.8 accuracy Same as (549+264)/1000 (from diagonal of confusion matrix)
(549+264)/1000

# classification=accuracy, while for regressions is R2 and RMSE



# Q: DOES IT MAKE SENSE TO SPLIT THE TRAINING DATA IN TWO PARTS IN THE RF?
# Build forest directly with all training data since predictions are OOB




##### Predict for contest ########
# Change class also in tedata (earlier we did that for trdata)
tedata$pclass<-as.factor(tedata$pclass)
tedata$sex<-as.factor(tedata$sex)

prediction_RF_te_new <- predict(RF, newdata = tedata, type = "class")
sum(prediction_RF_te_new == 1)/length(prediction_RF_te_new) # RF
# Check the confusion matrix with real data
# 28% of people predicted to survive

# X axis is nr of trees
plot(RF$err.rate[,1],type="l") #OOB
plot(RF$err.rate[,2],type="l") #missclassified as dead (false-negative)
plot(RF$err.rate[,3],type="l") #missclassified as alive but dead in reality (false-positi)

plot(RF)
# Red Line: False negative Error rate
# Green Line: False positive Error rate (classified as alive but dead in reality)
# Black Line: OOB Error rate

# False negative < False positive: It is more difficult to predict
# death than survival


# OOB error initially is high, then drops and gets constant
# we cannot improve this error after about 100 trees

#### Tune mtry ####
?tuneRF 
names(trdata)
colnames(trdata)
# Subset columns because NA in predictors not allowed when tuning
tuneRF(trdata[,c(1,4,5,6,9,11)],trdata[,2], stepFactor = 3, ntree=300, improve=0.05) #at least 0.05 the improve of OOB error to continue
# stepfactor: at each iteration mtry is inflated or deflated by this value

# Distrib. of nr of nodes: 100 trees contain the majority of the nodes
hist(treesize(RF))
?treesize


#### Var imp ####
# Variable importance plots: How does the model perform without one of each vars?
# if we remove one var, which will be the MeanDecrease in Accuracy
# i.e., the mean increase in MCE error
# big value = max importance in contributing in accuracy

# Gini= node impurity measure=tot variance across nodes: if 50% obs in every node
# --> high node impurity --> no very informative split
# Gini= how pure are the nodes at the end of the tree without each var
# by how much Gini decreases if we remove one variable
varImpPlot(RF, sort=T)

?importance
# total decrease in node impurities from splitting on the variable, averaged over all trees

importance(RF, type=1) # mean decrease in accuracy
importance(RF, type=2) # mean decrease in node impurity
# Note: neg. importance means no useful to predict y (do not interpret literally)

# type=1: Take MCE OOB, and take the MCE OOB w/o one predictor. Do the same
# for all trees where this predictor is contained. The difference between the two are then averaged over all trees, and normalized by the standard 
# deviation of the differences.

# type=2: is the total decrease in node impurities from splitting on the variable, 
# averaged over all trees. For classification, the node impurity is measured by the Gini index. 
# For regression, it is measured by residual sum of squares.

# More details on MCE and Gini, e.g., here https://www.bogotobogo.com/python/scikit-learn/scikt_machine_learning_Decision_Tree_Learning_Informatioin_Gain_IG_Impurity_Entropy_Gini_Classification_Error.php

?varUsed 

# survived ~ pclass + age + sex + fare + embarked + sibsp
# Check original order (pclass  + age + sex + fare + embarked + sibsp)
varUsed(RF) # for values for each var
# so the second variable occured 31k times in the RF (note that a variable occurs more than ones in 
# a tree because it can be splitted upon in different branches of the tree)
# No. of variables tried at each split: 2
# mtry = 2 means that two variables are randomly selected in each SPLIT in each tree


#### Marginal Effect Plots ####
# Partial dependence plot --> Marginal effects
# shows the marginal effect one or two features have on the predicted outcome 
?partialPlot
# First estimate the forest
# Marginal effect of x1? Take different values of x1
# Predict outcome for all individuals for that given value of x1
# Average all predictions for that level of x1
# Rescale y-axis s.t. 0 means that we have the same prediction for different values of x1

# The partial function tells us for given value(s) of feature x1 what the average marginal effect on the prediction is

# Negative values (in the y-axis) mean that the corresponding
# x1 value is associated to low probability to observe the reference class, ceteris paribus

# Plots are rescaled to be around 0
# To allow to compare plots of different variables

# More info: https://christophm.github.io/interpretable-ml-book/pdp.html

# Here reference class is death
partialPlot(RF, pred.data=trdata,
            x.var="fare")
# the highest predictive power is between 0 and 100
# after fare= 200, then it doesnt help much more in predicting survival

# marg effect on probability to survive (positive=higher probability)
partialPlot(RF, pred.data=trdata, which.class=1,
            x.var="age")

# marg effect on probability to die (positive=higher probability)
partialPlot(RF, pred.data=trdata, which.class=0,
            x.var="age")


partialPlot(RF, pred.data=trdata,
            x.var="sex") # being male helps more predicting to live or not
# if you are male you more likely died
# if you are female you more likely survived but there is higher heterogeneity across females
# prob of survival are more heterogeneous among females (e.g. depends by pclass and social status)

partialPlot(RF, pred.data=trdata,
            x.var="pclass") #3rd class has the highest predictive power


# Extract tree from RF
getTree(RF, 1, labelVar = T) #get the first tree
# status = -1 we are in the terminal node and the classification is the last one
# at status -1 we have no left/right daughter



# Compute partial dependence data for age and pclass
library(pdp)
pd <- partial(RF ,pred.var = c("age"), plot = TRUE,
              plot.engine = "ggplot2", rug=TRUE)
# Rug=TRUE are one-dimensional plots added to the axes.
# displaying min-max and deciles of the true distribution of x in the training data
# Other predictions are made with extrapolated values of x
pd

# More beautiful pdp plots

# lattice-based PDP
library(pdp)
library(dplyr)
p1 <- RF %>%  # the %>% operator is read as "and then"
  partial(pred.var = "age") %>%
  plotPartial(smooth = TRUE,  ylab = expression(pf(age)),
              main = "Marginal effect of age on probability to die (ref. class)")

p1

p1class <- RF %>%  # the %>% operator is read as "and then"
  partial(pred.var = "pclass") %>%
  plotPartial(smooth = F,  ylab = expression(pf(age)),
              main = "Marginal effect of class on probability to die (ref. class)")

p1class

p1sex <- RF %>%  # the %>% operator is read as "and then"
  partial(pred.var = "sex") %>%
  plotPartial(smooth = TRUE,  ylab = expression(pf(age)),
              main = "Marginal effect of sex on probability to die (ref. class)")

p1sex

# ggplot-based PDP
p2 <- RF %>%  # the %>% operator is read as "and then"
  partial(pred.var = "age") %>%
  autoplot(smooth = TRUE, ylab = expression(pf(age))) +
  theme_light() +
  ggtitle("ggplot2-based PDP")

p2 

library(gridExtra)
grid.arrange(p1, p2, ncol = 2)  


# MULTIPLE PREDICTORS
pd <- partial(RF, pred.var = c("age", "pclass"))
# Default PDP
pdp1 <- plotPartial(pd, smooth=TRUE, ylab = expression(pf(age, class)))
pdp1

# or

# MULTIPLE PREDICTORS
pdp1 <- RF %>% 
  partial(pred.var = c("age", "pclass")) %>% 
plotPartial()

pdp1

# Add contour lines and use a different color palette
rwb <- colorRampPalette(c("red", "white", "blue"))
pdp2 <- plotPartial(pd, contour = TRUE, col.regions = rwb)
pdp2



# 3-D surface
pdp3 <- plotPartial(pd, levelplot = FALSE, zlab = "fare", colorkey = TRUE, 
                    screen = list(z = -20, x = -60))
pdp3
# Figure 5
grid.arrange(pdp1, pdp2, pdp3, ncol = 3)




# Only in the convex hull = it outlines the region of the predictor space that the model was trained on
# Examples of PDPs restricted to the convext hull of the features of interest 
p1 <- partial(RF, pred.var = c("pclass", "age"), plot = TRUE, chull = TRUE)
p2 <- partial(RF, pred.var = c("pclass", "age"), plot = TRUE, chull = TRUE,
              palette = "magma")
grid.arrange(p1, p2, nrow = 1)  # Figure 7


##### Proximity and Multid-scaling plot ######

# Multidim-scaling plot (two classes, two colors)
RF$proximity
unique(RF$proximity)
# The proximity in Random Forests is defined to be the proportion, taken over all the trees in the forest,
# of the times that two observations end up in the same terminal node.

# Check original order pclass  + age + sex + fare + embarked + sibsp
# This is helpful in identifying multivariate outliers if there are any.
# The proximity matrix is used as input to the classical scaling algorithm.
MDSplot(RF,k=2, trdata$survived) 
# It just tells you how far apart (relatively) these clusters are from each other. 
# Pairs of objects that are very similar will have large values (close
# to 1) and pairs of objects that are very dissimilar will have small values (close to 0).

# Mostly positive values means that lots of points are similar in dim 1 and dim 2
# identifies hree well-defined clusters:
# one very similar in dim 2 but VERY dissimilar in dim 1 (top left)
# one very similar in dim 1 but QUIET dissimilar in dim 2 (bottom right) 
# one very similar in dim 1 AND in dim 2 (top right)
MDSplot(RF,k=4, trdata$survived)
?MDSplot


#### Permutation feature importance ####
# indicative of how much the model depends on the feature. 
# = change in predictions when a single feature value is randomly shuffled
# This procedure breaks the relationship between the feature and the target
# This technique benefits from being model agnostic and can be calculated many times with different permutations of the feature.

#### --> EDARF ####
# Replication exercise: http://cran.nexr.com/web/packages/edarf/vignettes/edarf.html

#devtools::install_github("zmjones/edarf", subdir = "pkg")
library(edarf)
# supports party (cforest), randomForest, and randomForestSRC (rfsrc).
?partial_dependence
?variable_importance
?extract_proximity
?plot_pd
?plot_prox
?marginalPrediction

varimp_edarf<-variable_importance(RF, data=trdata, nperm=2)
plot_imp(varimp_edarf)
# aggregate permutation importance:
# is the mean difference between the original predictions and 
# the predictions made when one var is permuted

#' @article{jones2016,
#'   doi = {10.21105/joss.00092},
#'   url = {http://dx.doi.org/10.21105/joss.00092},
#'   year  = {2016},
#'   month = {oct},
#'   publisher = {The Open Journal},
#'   volume = {1},
#'   number = {6},
#'   author = {Zachary M. Jones and Fridolin J. Linder},
#'   title = {edarf: Exploratory Data Analysis using Random Forests},
#'   journal = {The Journal of Open Source Software}
#' }

################# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ####################################

#### EXTRA. Use trees to replace NAs ####

# Replace some real data with NAs
trdataNA <- trdata
# Randomly take 16 rows
trdataNA$age[tr[1:15]] <- NA
class(trdataNA$age)

# "method":  if y is a factor then method = "class" is assumed, otherwise method = "anova" is assumed

agefit <- rpart(age ~ pclass + sex + sibsp + parch + fare + embarked,
                data= trdataNA[!is.na(trdataNA$age),], 
                method="anova")

# NAs
trdataNA$age[is.na(trdataNA$age)] 
# true data
trdata$age[is.na(trdataNA$age)] 
# predicted data
predict(agefit, trdataNA[is.na(trdataNA$age),]) # predicted data

# Compare to true data --> misprediction in %:
(trdata$age[is.na(trdataNA$age)]-predict(agefit, trdata[is.na(trdataNA$age),]))/trdata$age[is.na(trdataNA$age)]

# MSPE
mean((trdata$age[is.na(trdataNA$age)]-predict(agefit, trdata[is.na(trdataNA$age),]))/trdata$age[is.na(trdataNA$age)])
# Good predictions!

# Replace NAs
trdataNA$age[is.na(trdataNA$age)] <- predict(agefit, trdataNA[is.na(trdataNA$age),])
trdataNA$age[tr[1:15]] # replaced NAs


################# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ####################################

#### 6. Logistic Regression ####
str(trdata) # library(nnet)
# Logistic Regr. Model
logit<-multinom(survived ~ pclass  + age + sex + fare + embarked + sibsp, data = trdata[tr,])
summary(logit)

# Create result table in the slides
z <- summary(logit)$coefficients/summary(logit)$standard.errors
p <- (1 - pnorm(abs(z), 0, 1))*2 # we are using two-tailed z test

Pclass2 <- rbind(summary(logit)$coefficients,summary(logit)$standard.errors,z,p)
rownames(Pclass2) <- c("Coefficient","Std. Errors","z stat","p-value")
knitr::kable(Pclass2) # enlarge the Console window to see this
library(xtable)
xtable(Pclass2[c(1:2,4), c(1:5,9)])

# Prediction for test set
# Confusion matrix
p_test<-predict(logit,trdata[te,])
tab_test<-table(p_test, trdata[te,]$survived)
sum(diag(tab_test))/sum(tab_test) #accuracy = correct classification
# 0.78666 accuracy OOS

# Prediction for contest
p_tedataLOG<-predict(logit,tedata)
dim(tedata)
# Predicted share of survivors
sum(p_tedataLOG==1)/189

# IS prediction: Confusion matrix
p<-predict(logit,trdata[tr,])
tab<-table(p, trdata[tr,]$survived)
# 68 ppl who died but were classified as alive --> false-positive
sum(diag(tab))/sum(tab) #accuracy = correct classification
# 0.78 accuracy IS


# Benchmark
table(trdata[tr,]$survived)
415/700 #if accuracy is less than this, then do not use logistic model

# Model Performance Evaluation
pred<-predict(logit, trdata[tr,], type="prob")
head(pred) # 1st value= prob that first obs survived
# first element: prediction is survived
# second element: prediction is died
# third element: prediction is died
# pred. is 1 if prob is > 0.5
head(trdata[tr,"survived"]) # indeed first obs survived


# Note: using different prediction cutoff than 0.5 we get different MCE

# We could increase the prob. cutoff for which we assign a predicted prob. to
# be 0 (dead) or 1 (survived)
pred2<-prediction(pred, trdata[tr,]$survived)
eval<-performance(pred2, "acc")
plot(eval) # plots accuracy levels (1-MCE) at different prob.cutoffs
# around prob. of survival 0.6 we have the best accuracy of prediction
# We can decide to define as survived all with predicted probabilities >0.60
# and dead those with predicted probabilities <=0.60

# Get best cutoff prob. for prediction IS
eval # contains y.values
max<-which.max(slot(eval,"y.values")[[1]])
acc<-slot(eval, "y.values")[[1]][max] #opt. accuracy given opt. cutoff prob
cut<-slot(eval, "x.values")[[1]][max]
print(c(Accuracy=acc, Cutoff=cut))
# cutoff chosen is 0.66 
# corresponding accuracy is 0.80
# It represents an improvement compared to before
# vs. 0.5 prob. cutoff with less accuracy of 0.78
sum(diag(tab))/sum(tab)

# MCE now is lower than before without cutoff tuning

# We may be more concerned in predict survivals with accuracy than deaths
# --> use ROCR
# tpr: True positive rate. P(Yhat = + | Y = +). Estimated as: TP/P.
# fpr: False positive rate. P(Yhat = + | Y = -). Estimated as: FP/N

tab # true positive rate is 199/(199+86) = acc level to predict 1 (look at 1 column)
# false prediction rate is 68/(347+68) is false positive (look at 0 column)

# Check model performance
# We use AUC (Area Under The Curve) ROC (Receiver Operating Characteristics) curve. It is one of the 
# most important evaluation metrics for checking any classification model's performance. 
pred2<-prediction(pred, trdata[tr,]$survived)
roc<-performance(pred2, "tpr", "fpr") #true positive rate= sensitivity, false pos rate = 1-specificity
plot(roc)
plot(roc, colorize=T, ylab="sensitivity",xlab="1-specificity=false positive")
# ideally we would like to have AUC area =1
# or a inverse-L ROC curved
# At any level of FPR, we get TPR=1
# At any level of TPR, we get FPR=0
abline(a=0,b=1)
# AUC area below this line is 0.5
# Look whether model does better than this line 
# Whether ROC curve is above (or AUC area>0.5)
# Is the ROC line above this? If yes, then is good performance

# AUC: Find area under the ROC curve: rectangular area = 1
# area below line is 0.5, the greater the AUC area the better
# Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s.
auc<-performance(pred2, "auc")
auc<-unlist(slot(auc, "y.values"))
auc<-round(auc,4)
legend(.6,.8,auc,title="AUC")
# AUC is 0.8, it means there is 80% chance that model will be able to 
# distinguish between the two classes.

# When AUC is approximately 0.5, model has no discrimination capacity 
# to distinguish between positive class and negative class.
# = ideal measure of separability between classes

# Check here for visual representation:
# https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5




################# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ####################################

##### Compare predictions ################
###### IS prediction accuracy 
# (IS accuracy, per se, not interesting)
sum(trdata[tr,]$survived == prediction_IS)/700 # 0.981
sum(trdata[tr,]$survived == prediction_IS_pruned)/700 # 0.847

# After adding/creating new variables
sum(trdata[tr,]$survived == prediction_IS2)/700 # 0.981
sum(trdata[tr,]$survived == prediction_IS2_pruned)/700 # 0.858

sum(diag(tab))/sum(tab) #accuracy LOGIT 0.78

# OOS or Real accuracy measure we should look at
# OOS prediction accuracy
sum(trdata[te,]$survived == prediction0)/length(te) # unpruned 0.7266
sum(trdata[te,]$survived == prediction1)/length(te) # pruned 0.7966

# After adding/creating new variables
sum(trdata[te,]$survived == prediction2)/length(te) # unpruned with more variables 0.7266
sum(trdata[te,]$survived == prediction3)/length(te) # pruned with more variables 0.80
# last model predicts the best OOS
# we expect the same for predictions on any new data

# RF:
sum(trdata[te,]$survived == prediction_RF_te, na.rm = T)/length(te) # 0.85
# or, OOB for all 1000 obs.
sum(trdata$survived == prediction_RF_te, na.rm = T)/1000# 0.813

sum(diag(tab_test))/sum(tab_test) #accuracy LOGIT 0.78666


##### Survivors of unknown dataset ################

# Share of predicted survivors for unknown dataset
sum(predictionbo0 == 1)/length(predictionbo0) # unpruned 0.402
sum(predictionbo1 == 1)/length(predictionbo1) # pruned 0.269

# After adding/creating new variables
sum(predictionbo2 == 1)/length(predictionbo2) # unpruned more variables 0.396
sum(predictionbo3 == 1)/length(predictionbo3) # pruned more variables 0.27

# RF:
sum(prediction_RF_te_new == 1)/length(prediction_RF_te_new) # RF

sum(p_tedataLOG==1)/length(p_tedataLOG) # LOGIT


################# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ####################################


########## ---> TRUE DATASET (to be shared in the future) ##########
######### True dataset to compare your predictions with truth
#setwd("../../../seadrive_root/Marica V/My Libraries/Teaching_WS22/ML/R/trees")
#setwd("/Volumes/Extreme SSD/UIBK/Semester von PC/UBIK/Semester 4/VU Aktuelle Entwicklungen in Wirtschaft und Gesellschaft Maschinelles Lernen für Progn/R/data")
#/Volumes/Extreme SSD/UIBK/Semester von PC/UBIK/Semester 4/VU Aktuelle Entwicklungen in Wirtschaft und Gesellschaft Maschinelles Lernen für Progn/R/data)
setwd("/Volumes/Extreme SSD/UIBK/Semester von PC/UBIK/Semester 4/VU Aktuelle Entwicklungen in Wirtschaft und Gesellschaft Maschinelles Lernen für Progn/R/data")

titdata<-read.csv2("titanic-passengers.csv", header = TRUE, sep=";",dec=".")
head(titdata)
dim(titdata)
unique(tedata$name)

# Check survival only for passengers in test data
checkdata <- titdata[titdata$Name %in% tedata$name, ]

# Transform survived variable No-Yes into 0-1
levels(checkdata$Survived) <- c("0", "1")
checkdata$name <- checkdata$Name
checkdata$name %in% tedata$name

tedata_all <- merge(checkdata[, c("Survived", "name")], tedata, by=c("name"), all.y=T)
colnames(tedata_all)
dim(tedata_all)
head(tedata)
tedata_all$Survived <- ifelse(tedata_all$Survived=="Yes",1,0)


### ### ### ### ###
### PREDICTIONS ###
### ### ### ### ###

# TRUE OUTCOME: Best pred. accuracy
# PRED ACCURACY: Pruned tree performs the best
sum(tedata_all$Survived == predictionbo0)/189 # unpruned 0.54
sum(tedata_all$Survived == predictionbo1)/189 # pruned 0.62
sum(tedata_all$Survived == predictionbo2)/189 # unpruned with more variables 0.54
sum(tedata_all$Survived == predictionbo3)/189 # pruned with more variables 0.62

sum(tedata_all$Survived == prediction_RF_te_new)/189 # RF 0.61
sum(tedata_all$Survived == p_tedataLOG)/189 # Logit 0.58

### ### ### #
### TRUTH ###
### ### ### #

# Compare the SHARE of predicted survivors 
sum(tedata_all$Survived)/189 # 31.2% survived

### ### ### ### ###
### PREDICTIONS ###
### ### ### ### ###

# Pruned tree with more variables gets the closest share
sum(predictionbo0 == 1)/length(predictionbo0) # unpruned
sum(predictionbo1 == 1)/length(predictionbo1) # pruned

sum(predictionbo2 == 1)/length(predictionbo2) # unpruned more variables 0.396
sum(predictionbo3 == 1)/length(predictionbo3) # pruned more variables 0.27

# RF:
sum(prediction_RF_te_new == 1)/length(prediction_RF_te_new) # RF 0.28

sum(p_tedataLOG==1)/length(p_tedataLOG) # LOGIT 0.306
# Logit is closest in terms of share of survivors

# Try RF with new variables!
# Try Logit with new variables!
