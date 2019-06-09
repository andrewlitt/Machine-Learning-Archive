%% CMPE 452 Assignment 1 - Implementation of a Perceptron 
% Andrew Litt
% 10150478, 14asl

% Requires the PerceptronTrain(...) and PerceptronTest(...) functions in
% the same directory

% Prints output to results.txt

%% Section 1: Parsing & Formatting Datasets
% Open & parse TRAINING dataset
fileID = fopen('train.txt');
data = textscan(fileID,'%f %f %f %f %s','Delimiter',',');
trainData = [data{1},data{2},data{3},data{4}];
trainAns = string(data{5});
fclose(fileID);

% Map the training set answers to binary outputs
tmp = zeros(length(trainAns),3);
for i = 1:length(trainAns)
    switch trainAns(i)
        case 'Iris-setosa'
            tmp(i,:) = [1, 0, 0];
        case 'Iris-versicolor'
            tmp(i,:) = [0, 1, 0];
        case 'Iris-virginica'
            tmp(i,:) = [0, 0, 1];
    end
end
trainAns = tmp;

%Open & parse TESTING dataset
fileID = fopen('test.txt');
data = textscan(fileID,'%f %f %f %f %s','Delimiter',',');
testData = [data{1},data{2},data{3},data{4}];
testAns = string(data{5});
testAnsString = testAns; % savea copy for printing later
fclose(fileID);

% Map the testing set answers to binary outputs
tmp = zeros(length(testAns),3);
for i = 1:length(testAns)
    switch testAns(i)
        case 'Iris-setosa'
            tmp(i,:) = [1, 0, 0];
        case 'Iris-versicolor'
            tmp(i,:) = [0, 1, 0];
        case 'Iris-virginica'
            tmp(i,:) = [0, 0, 1];
    end
end
testAns = tmp;

%Invert datasets for correct formatting in following functions
trainData = trainData';
trainAns = trainAns';
testData = testData';
testAns = testAns';

%% Section 2: Train & Testing Perceptrons using built functions
% Approach - train 2 perceptrons to classify the SETOSA and 
% VIRGINICA species. Should both return 0, classify as VERSICOLOR.

% Train & Test SETOSA Perceptron
learnRate = 0.1;     
maxIterS = 1000;
range = 5;
wi_setosa = -range + 2*range*rand(5,1);
% Call training & test functions
[wf_setosa,setosaIter] = PerceptronTrain(trainData,trainAns(1,:),wi_setosa,learnRate,maxIterS);
[y_setosa, e_setosa] = PerceptronTest(testData,testAns(1,:),wf_setosa);
% Calculate precision, recall & error
setosa_precision = sum(y_setosa(1:10))/sum(y_setosa);  % True Positives / (True Positives + False Positives)
setosa_recall = sum(y_setosa(1:10))/sum(testAns(1,:)); % True Positives / (True Positives + False Negatives)
setosa_error = sum(e_setosa);

% Train & Test VIRGINICA Perceptron
learnRate = 0.05;     
maxIterV = 3000;
range = 4;
wi_virginica = -range + 2*range*rand(5,1);
% Call training & test functions
[wf_virginica,virginicaIter] = PerceptronTrain(trainData,trainAns(3,:),wi_virginica,learnRate,maxIterV);
[y_virginica, e_virginica] = PerceptronTest(testData,testAns(3,:),wf_virginica);
% Calculate precision, recall & error
virginica_precision = sum(y_virginica(21:30))/sum(y_virginica);
virginica_recall = sum(y_virginica(21:30))/sum(testAns(3,:));
virginica_error = sum(e_virginica);

% Should both perceptrons return 0, classify point as VERSICOLOR
y_versicolor = zeros(1,length(y_setosa));
for i = 1:length(y_setosa)
    if(y_setosa(i) == 0 & y_virginica(i) == 0)
        y_versicolor(1,i) = 1;
    end
end
% Calculate precision, recall & error
versicolor_precision = sum(y_versicolor(11:20))/sum(y_versicolor);
versicolor_recall = sum(y_versicolor(11:20))/sum(testAns(2,:));
versicolor_error = abs(10 - sum(y_versicolor));
% Join arrays: 1 = SETOSA, 2 = VERSICOLOR, 3 = VIRGINICA
y = (y_setosa + 2*y_versicolor + 3*y_virginica)';
total_error = sum(setosa_error) + sum(virginica_error) + sum(versicolor_error);

%% Section 3: Use MATLAB's Built-In Tools with the Same Approach
% MATLAB Built-in Perceptron tool
% If you try to make it classify all 3, it sucks. So, we'll take the same
% approach of classifying VERSICOLOR by process of elimination
net = perceptron;
net = train(net,trainData,[trainAns(1,:);trainAns(3,:)])
Y = sim(net,testData);
Y = [Y(1,:); zeros(1,length(Y)); Y(2,:)]; %Reformat to add a 2nd row
for i = 1:length(Y)
    if(Y(1,i) == 0 & Y(3,i) == 0)
        Y(2,i) = 1;
    end
end % Classifying VERSICOLOR Points

msetosa_precision = sum(Y(1,1:10))/sum(Y(1,:));  
msetosa_recall = sum(Y(1,1:10))/sum(testAns(1,:));

mvirginica_precision = sum(Y(2,11:20))/sum(Y(2,:));
mvirginica_recall = sum(Y(2,11:20))/sum(testAns(2,:));

mversicolor_precision = sum(Y(3,21:30))/sum(Y(3,:));
mversicolor_recall = sum(Y(3,21:30))/sum(testAns(3,:));

Y = (Y(1,:) + 2*Y(2,:) + 3*Y(3,:))';

%% Section 4: Output Data File
%Parse answer arrays back into strings
tmp = strings(length(y),1);
for i = 1:length(y)
    switch y(i)
        case 1
            tmp(i,:) = 'Iris-setosa';
        case 2
            tmp(i,:) = 'Iris-versicolor';
        case 3
            tmp(i,:) = 'Iris-virginica';
    end
end
y = tmp;

tmp = strings(length(Y),1);
for i = 1:length(Y)
    switch Y(i)
        case 1
            tmp(i,:) = 'Iris-setosa';
        case 2
            tmp(i,:) = 'Iris-versicolor';
        case 3
            tmp(i,:) = 'Iris-virginica';
    end
end
Y = tmp;

fileId = fopen('results.txt','w');
fprintf(fileID,'| Actual            | My Perceptron Guess | MATLAB Tool Guess       | \r\n');
fprintf(fileID,'|-------------------|---------------------|-------------------------|\r\n');
for i = 1:length(y)
    % for some reason a normal fprintf was mixing up the indexes  
    % while printing, so I just looped through the array
    fprintf(fileID,'| %17s | %19s | %23s |\r\n',testAnsString(i),y(i),Y(i));
end
fprintf(fileID,'|-------------------|---------------------|-------------------------|\r\n\n');
fprintf(fileID,'Analysis of Implemented Perceptron\r\n\n');
fprintf(fileID,'SETOSA initial weight vector:    Wi= %.2f %.2f %.2f %.2f %.2f\r\n',wi_setosa);
fprintf(fileID,'SETOSA final weight vector:      Wf= %.2f %.2f %.2f %.2f %.2f\r\r\n\n',wf_setosa);
fprintf(fileID,'VIRGINICA initial weight vector: Wi= %.2f %.2f %.2f %.2f %.2f\r\n',wi_virginica);
fprintf(fileID,'VIRGINICA final weight vector:   Wf= %.2f %.2f %.2f %.2f %.2f\r\r\n\n',wf_virginica);
fprintf(fileID,'TOTAL classification error: E = %d\r\r\n\n', total_error);
fprintf(fileID,'SETOSA iterations:    %d/%d limit\r\n',setosaIter,maxIterS);
fprintf(fileID,'VIRGINICA iterations: %d/%d limit\r\n',virginicaIter,maxIterV);
fprintf(fileID,'(If perceptron is below limit, it converged; else, it terminated with its final weight)\r\r\n\n',virginicaIter,maxIterV);
fprintf(fileID,'SETOSA     Precision: %.3f, Recall: %.3f\r\n',setosa_precision,setosa_recall);
fprintf(fileID,'VERSICOLOR Precision: %.3f, Recall: %.3f\r\n',versicolor_precision,versicolor_recall);
fprintf(fileID,'VIRGINICA  Precision: %.3f, Recall: %.3f\r\r\n\n',virginica_precision,virginica_recall);
fprintf(fileID,'-----------------------------------------------\r\n');
fprintf(fileID,'Analysis of MATLAB Perceptron\r\n\n');
fprintf(fileID,'SETOSA Precision: %.3f, Recall: %.3f\r\n',msetosa_precision,msetosa_recall);
fprintf(fileID,'VERSICOLOR Precision: %.3f, Recall: %.3f\r\n',mversicolor_precision,mversicolor_recall);
fprintf(fileID,'VIRGINICA Precision: %.3f, Recall: %.3f\r\n',mvirginica_precision,mvirginica_recall);
fclose(fileID);