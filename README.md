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

