# Random Forest Classifier

# Data Preprocessing
# Import Dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding Target Feature As Factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Spliting Training and Test Set 
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scalling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])


# Modeling
# Fitting Random Forest Classification on Training Set
library(randomForest)
classifier = randomForest(x = training_set[-3],
                          y = training_set$Purchased,
                          ntree = 10)

# Predicting With Random Forest Classifier
y_pred = predict(classifier, newdata = test_set[-3])

# confusion Matrix
cm = table(test_set[, 3], y_pred)


# Visualisation
# Visualising Random Forest Classifier on Training Set
library(ElemStatLearn)
set = training_set
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Random Forest Classifier (Training Set)',
     xlim = range(x1), ylim = range(x2),
     xlab = 'Age', ylab = 'EstimatedSalary')
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_pred == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# Visualising Random Forest Classifier on Test Set
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, grid_set)
plot(set[, -3], main = 'Random Forest Classification (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Choosing Number Of Trees
plot(classifier)