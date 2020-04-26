# ASL
## 1. Introduction
In the presence of ordinal target variables, traditional applications of machine learning classifiers would treat
each class as nominal values, therefore those standard classification algorithms cannot utilize the advantage of
inherent ordering information. In this homework, I seek to implement a simple approach, presented by Eibe. F
and Mark. H (2001), that enables standard classification algorithms to be benefit from ordering information in
ordinal target variables. Generally, the idea is to build 2 models each predicting the probability of a data point
being greater than a class value, then integrate model results to get prediction. The experiment result indicates
that the ensemble model outperforms naive classification approach, which treats each class as a set of unordered
values.

## 2. Existing work
Packages to solve ordinal regression problems developed over the last years, such as OCAPIS (Scala), ORCA
(Matlab), Mord (python) and Ordinal (R). However, some of those packages are either only implemented in
“not super popular” platforms, such as OCAPIS in Scala or ORCA in Matlab, which may require programmer
to have prior knowledge about those platforms; or capacity of models inherit in those packages are relatively
low (such as Mord), and have difficulties in handling high dimensional data (Some of them cannot even beat
standard classifiers, which simply treat y as nominal variables).

This simple method could be easily implemented in Python and R, which we are relatively more familiar. In this
way, we can build ordinal classifiers and tune hyper parameters in a traditional “machine learning” way, and
take the advantage of most classifiers embedded , as long as the classifier outputs class probability (such as
neural network and random forest).

## 3. Methodology
To take advantages of ordering information stored in target variable (wine quality), I convert the 3–class ordinal
regression into 2 binary classification problem.

Firstly, based on the original ordinal target variable, two binary target variables, V1 and V2 are created to
represent if the current quality is greater than the quality which the field represents, then the new binary target
equals to 1,else 0. For example, the binary target (V1) is 1 if original ordinal target (wine quality)> 1. The
following table shows the outputs of transformation.

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/ASL/master/images/ASL1.png "Optional title")

Once the target variable transformation is done, each binary target variable (V1, V2) is fitted by one standard
classification model. After model fitting, each of these models will be able to predict the probability of current
quality being greater than a quality which the model is built for. The following graph show an example of such
model fitting process.

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/ASL/master/images/ASL2.png "Optional title")

After we trained 2 binary models, the probability of current quality belonging to each class (wine quality) for
one record (row) can be calculated respectively, using the following formula:

 · Pr(y=1) = 1-Pr(Target > 1)
 · Pr(y=2) = Pr(Target>1)-Pr(Target > 2)
 · Pr(y=3) = Pr(Target > 2)
 
Once we get value of Pr(y = 1/2/3), the quality which has the highest probability to the current record was
selected as the predicted quality of the current fold.

Please noted that since I didn’t specify any classifier in the above method, we could actually implement any
“machine learning classifier” to calculate the class probability Pr(y=1/2/3).
 
## 4. Experiment design
After feature scaling, firstly the grid search was implemented to train standard Randomforest classifier and
Mutil layer classifier, by simply treating y as nominal variable. After that, we convert the 3–class ordinal
regression into 2 binary classification problem, and proceed ordinal Randomforest classifier and ordinal MLP
classifier, using hyper parameters we got from last step (the best hyper parameter returned by doing grid search
in Standard Randomforest classifier and Mutil layer classifier). When the training of Ordinary models was
done, we train LogisticIT and LogisticAT models by using Mord package. Finally, we implement Repeated
Stratified KFold(n_splits=5, n_repeats=10) to get the score of each model.

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/ASL/master/images/ASL3.png "Optional title")

There are 6 models was compared in this homework, and the description of them can be seen from the
following table:

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/ASL/master/images/ASL4.png "Optional title")

## 5. Experiment results and conclusion
In this homework, I compare the performance of 6 models in the wine quality 3-class ordinal regression
problem. Those models’ performance can be seen from the following table:

![Alt text](https://raw.githubusercontent.com/chaoyangzhengnash/ASL/master/images/ASL5.png "Optional title")

From the above table, we noticed that after transforming our ordinal classification problems to 2 binary
classification problems, scores of both standard classifiers (Random forest classifier and multi-layer classifier)
have improved. Meanwhile, we noticed that the ordinal Multi-layer classifier has the best score after doing
repeated stratified cross validation, therefore, we choose the ordinal Multi-layer classifier(1) as our model and
run it on the test dataset to get prediction.

## 6. Critical thinking
One of the major flaw of this approach is that since each classifier are trained independently, each of them are
independent from each other, the method won’t give the true probability as well (it won’t sum up to 1)
