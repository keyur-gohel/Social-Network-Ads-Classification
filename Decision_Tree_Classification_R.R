# Decision Tree Classification

# Data Preprocessing
# Import Dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding Targer Feature As Factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Spliting Training and Test
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])


# Modeling
# Fitting Deision Tree Classifier to Training Set
#install.packages('rpart')
library(rpart)
classifier = rpart(formula = Purchased ~ .,
                   data = training_set)

# Predicting With Decision Tree Classifier
y_pred = predict(classifier, newdata = test_set[-3], type = 'class')

# Confusion Matrix
cm = table(test_set[, 3], y_pred)


# Visualisation
# Visualising Decision Tree Classifier on Training set
library(ElemStatLearn)
set = training_set
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3],
     main = 'Decision Tree Classifier (Training Set)',
     xlim = range(x1), ylim = range(x2),
     xlab = 'Age', ylab = 'EstimatedSalary')
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualising Decision Tree Classifier on Test set
library(ElemStatLearn)
set = test_set
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set, type = 'class')
plot(set[, -3],
     main = 'Decision Tree Classifier (Training Set)',
     xlim = range(x1), ylim = range(x2),
     xlab = 'Age', ylab = 'EstimatedSalary')
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Ploting Tree (only Possible Without Feature Scaling)
plot(classifier)
text(classifier)