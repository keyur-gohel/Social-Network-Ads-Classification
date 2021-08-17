# Kernel SVM
# Data Preprocessing

# Import Dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding Targer Feature As Factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Spliting Training and Test Set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_Set = subset(dataset, split == TRUE)
test_Set = subset(dataset, split == FALSE)

# Feature Scalling
training_Set[-3] = scale(training_Set[-3])
test_Set[-3] = scale(test_Set[-3])


# Modeling
# Fitting Kernel SVM to Training Set
library(e1071)
classifier = svm(formula = Purchased ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting With kernel SVM
y_pred = predict(classifier, newdata = test_Set[-3])

# Confusion Matrix
cm = table(test_Set[, 3], y_pred)


# Visualisation
# Visualising With Training Set
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising Kernel SVM on Test Set
library(ElemStatLearn)
set = test_set
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'Kernel SVM (Tarining Set)',
     xlab = 'Age', ylab = 'EstimatedSalary',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))
