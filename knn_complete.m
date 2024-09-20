clear all
clc
close all

% Import the phone data
d = importdata('phonedata.txt'); % Use this if the file contains text data

% Convert the text data into numbers
[m, n] = size(d.data);
X = zeros(m, 1); % Initialize X to store gender (1 for male, 0 for female)

% Extract gender information from textdata
for i = 1:m
    a = d.textdata{i}; % Get the text data (gender information)
    len = length(a);   % Get the length of the string
    if len == 4 % Assuming 'Male' has length 4
        X(i, 1) = 1; % 1 for male
    else
        X(i, 1) = 0; % 0 for female
    end
end

% Append numerical data (assumed to be present in columns 1 and 2)
X = [X d.data(:, 1:2)]; % Final feature matrix with all numerical data

% Normalizing features
X = featureNormalize(X); % Ensure featureNormalize is defined

% Target value
y = d.data(:, 3); % Assume column 3 is the target variable

% Split the dataset into training and testing sets
Xtrain = X(1:360, :);
ytrain = y(1:360);
Xtest = X(361:end, :);
ytest = y(361:end);

% KNN Algorithm
k = 3; % Define the number of neighbors

% Calculate predictions for the test data
ypred = knn(Xtrain, ytrain, k, Xtest); % Predict using KNN (Xtest is new data)

% Calculate confusion matrix, accuracy, precision, recall, F1 score
[Cmatrix, ACC, P, R, F1] = confusionmatrix(ypred, ytest); % Assuming confusionmatrix function is defined
disp('Confusion Matrix:');
disp(Cmatrix);
disp(['Accuracy: ', num2str(ACC)]);
disp(['Precision: ', num2str(P)]);
disp(['Recall: ', num2str(R)]);
disp(['F1 Score: ', num2str(F1)]);