%% CMPE 452 Assignment 2: Implement a Backpropogation Network
% Andrew Litt
% 10150478, 14asl

% Requires the computeActivation(...) function in the same directory
clc
clear all
% Prints output to results.txt

%% Appendix: Major System Parameters
epsilon = 0.1;
c = 0.9;
alpha = 0.4;
trainSize = 0.8;
validateSize = 0.1;% Partition of data to go to training
range = 0.1;

numHidden = 7;
numOut = 3;

%% Section 1: Parse Dataset & Perform Preprocessing

% Formatting: each column in matrix is a weight vector for the connections to 
% nodes of previous layer. So, dimensions are: 
% [Nodes in prev. layer x Nodes in current layer]

% Retrieve and split dataset into input data & answers
data = xlsread("assignment2Data.csv")';

[numFeatures, numPatterns] = size(data)

% Normalize Input Data
for f = 1:numFeatures-1
    maxFeature = 0;
    minFeature = inf;
    for p = 1:numPatterns
        if(data(f,p) > maxFeature)
            maxFeature = data(f,p);
        end
        if(data(f,p) < minFeature)
            minFeature = data(f,p);
        end
    end
    for p = 1:numPatterns
        data(f,p) = (data(f,p)-minFeature)/(maxFeature-minFeature);
    end
end

class1Data = [];
class2Data = [];
class3Data = [];

for i = 1:numPatterns
    switch data(12,i)
        case 5
            class1Data = [class1Data,data(:,i)];
        case 7
            class2Data = [class2Data,data(:,i)];
        case 8
            class3Data = [class3Data,data(:,i)];
    end
end

% Shuffle each dataset and trim to 175 points
class1Data = class1Data(:,randperm(size(class1Data,2)));
class2Data = class2Data(:,randperm(size(class2Data,2)));
class3Data = class3Data(:,randperm(size(class3Data,2)));

totalSize = 175;

class1Data = class1Data(:,1:175);
class2Data = class2Data(:,1:175);
class3Data = class3Data(:,1:175);

% Concatenate datasets and shuffle
equalDataset = [class1Data,class2Data,class3Data];
equalDataset = equalDataset(:,randperm(size(equalDataset,2)));
equalData = equalDataset(1:11,:);
equalAns = equalDataset(12,:);
clear data;
clear equalDataset;

% Map 5,7,8 outputs to binary vectors
tmp = zeros(3,length(equalAns));
for i = 1:length(equalAns)
    switch equalAns(i)
        case 5
            tmp(:,i) = [1-epsilon;epsilon;epsilon];
        case 7
            tmp(:,i) = [epsilon;1-epsilon;epsilon];
        case 8 
            tmp(:,i) = [epsilon;epsilon;1-epsilon];
    end
end
equalAns = tmp;

% Get idecies for training & validation sets
trainIndex = round(525*trainSize);
validateIndex = trainIndex + round(525*validateSize);

% Cut up training, validation & testing sets
trainData = equalData(:,1:trainIndex);
trainAns = equalAns(:,1:trainIndex);

validateData = equalData(:,trainIndex+1:validateIndex);
validateAns = equalAns(:,trainIndex+1:validateIndex);

testData = equalData(:,validateIndex+1:525);
testAns = equalAns(:,validateIndex+1:525);


[~, numValidatePatterns] = size(validateData);
[numFeatures, numTrainPatterns] = size(trainData);

% Defining the number of features, hidden nodes and output nodes
hiddenOutput = zeros(1,numHidden);
output = zeros(1,numOut);

% Initialize weight matricies with random values
hiddenWeights =  -range + 2*range*rand(numFeatures+1,numHidden); % weights TO hidden nodes
outputWeights = -range + 2*range*rand(numHidden+1,numOut); % weights TO output nodes
hiddenBiasIndex = numFeatures+1;
outputBiasIndex = numHidden+1;

% Initialize previous delta weight arrays for momentum use
hPrevDelta = zeros(numFeatures+1,numHidden);
oPrevDelta = zeros(numHidden+1,numOut);

deltaW = 0;
deltaojw21 = 0;

%% Section 2: Two Layer Network Algorithm

maxIter = 2000;
iter = 0;

prevTrainMSE = 0;
prevValidMSE = inf;
validMSE = inf;
% While below the epoch limit and above the min error
while (iter < maxIter && validMSE > 0.025)
    trainMSE = 0;
    validMSE = 0;
    % For each input pattern
    for p = 1:numTrainPatterns
        
        % Compute activations at hidden nodes
        for h = 1:numHidden
            hiddenOutput(h) = computeActivation(trainData(:,p),hiddenWeights(1:numFeatures,h),hiddenWeights(hiddenBiasIndex,h));
        end
        
        % Compute activations at output nodes & Mean Squared Error
        for o = 1:numOut
            output(o) = computeActivation(hiddenOutput',outputWeights(1:numHidden,o),outputWeights(outputBiasIndex,o));
            desired = trainAns(o,p);
            if(desired == (1-epsilon) && output(o) >= epsilon)
                e = 0;
            elseif(desired == epsilon && output(o) <= desired)
                e = 0;
            else
                e = desired - output(o);
            end
            trainMSE = trainMSE + e*e;
        end
        
        % Modify weights between hidden & output nodes
        for o = 1:numOut
            %Modify weights from each hidden node
            for h = 1:numHidden
                desired = trainAns(o,p);
                if(desired == (1-epsilon) && output(o) >= desired)
                    deltaW = 0;
                elseif(desired == epsilon && output(o) <= desired)
                    deltaW = 0;
                else
                    deltaW = c*hiddenOutput(h)*(desired-output(o))*output(o)*(1-output(o)) + alpha*oPrevDelta(h,o);
                    oPrevDelta(h,o) = deltaW;
                    outputWeights(h,o) = outputWeights(h,o) + deltaW;
                end
            end
            %Modify bias weight to each output node
            deltaW = c*1*(desired-output(o))*output(o)*(1-output(o)) + alpha*oPrevDelta(outputBiasIndex,o);
            oPrevDelta(outputBiasIndex,o) = deltaW;
            outputWeights(outputBiasIndex,o) = outputWeights(outputBiasIndex,o) + deltaW;
        end
        
        deltaojw21 = 0;
        %Calculate total error from the layer for use in next adjustments
        for o = 1:numOut
            for h = 1:numHidden+1
                desired = trainAns(o,p);
                if(desired == (1-epsilon) && output(o) >= desired)
                    deltaojw21 = deltaojw21 + 0;
                elseif(desired == epsilon && output(o) <= desired)
                    deltaojw21 = deltaojw21 + 0;
                else
                    deltaojw21 = deltaojw21 + (desired-output(o))*output(o)*(1-output(o))*outputWeights(h,o);
                end
            end
        end

        %Modify weights between input & hidden nodes
        for h = 1:numHidden
            for i = 1:numFeatures
                deltaW = c*deltaojw21*hiddenOutput(h)*(1-hiddenOutput(h))*trainData(i,p) + alpha*hPrevDelta(i,h);
                hPrevDelta(i,h) = deltaW;
                hiddenWeights(i,h) = hiddenWeights(i,h) + deltaW;
            end
            deltaW = c*deltaojw21*hiddenOutput(h)*(1-hiddenOutput(h))*1 + alpha*hPrevDelta(hiddenBiasIndex,h);
            hPrevDelta(hiddenBiasIndex,h) = deltaW;
            hiddenWeights(hiddenBiasIndex,h) = hiddenWeights(hiddenBiasIndex,h)+ deltaW;
        end
    end
    %DONE TRAINING FOR THIS EPOCH
    
    %Calculate MSE over VALIDATION SET
    for v = 1:numValidatePatterns
            % Hidden Activations
            for h = 1:numHidden
                hiddenOutput(h) = computeActivation(validateData(:,v),hiddenWeights(1:numFeatures,h),hiddenWeights(hiddenBiasIndex,h));
            end
            % Output Activations & Error Calc
            for o = 1:numOut
                output(o) = computeActivation(hiddenOutput',outputWeights(1:numHidden,o),outputWeights(outputBiasIndex,o));
                desired = validateAns(o,v);
                if(desired == (1-epsilon) && output(o) >= epsilon)
                    e = 0;
                elseif(desired == epsilon && output(o) <= desired)
                    e = 0;
                else
                    e = desired - output(o);
                end
                validMSE = validMSE + e*e;
            end 
    end
    validMSE = validMSE/(numValidatePatterns*numOut);
    trainMSE = trainMSE/(numTrainPatterns*numOut);
    
    % If the training error is shrinking, but the validation set
    % error is growing, then we are OVERFITTING the dataset and should
    % stop training the system.
    if(trainMSE < prevTrainMSE && validMSE > prevValidMSE && validMSE > trainMSE)
        iter = maxIter+1;
        disp('Stopping training due to Overfitting...');
    end
    fprintf('Iteration %d\n', iter);
    fprintf('trainMSE: %.10f\n',trainMSE);
    fprintf('validMSE: %.10f\n\n',validMSE);
    
    prevValidMSE = validMSE;
    prevTrainMSE = trainMSE;
    iter = iter + 1;
end

%% Section 3: Test Algoritm, Calculate Precision & Recall, Print Results to Sheet

falsePositives = zeros(1,numOut);
truePositives = zeros(1,numOut);
falseNegatives = zeros(1,numOut);
trueNegatives = zeros(1,numOut);


[numFeatures, numTestPatterns] = size(testData);
results = zeros(numTestPatterns+1,2,numOut);

disp('Testing...');
confTargets = zeros(1,numTestPatterns);
confOutputs = zeros(1,numTestPatterns);
for p = 1:numTestPatterns
    
    % Calculate the Output
    for h = 1:numHidden
        hiddenOutput(h) = computeActivation(testData(:,p),hiddenWeights(1:numFeatures,h),hiddenWeights(hiddenBiasIndex,h));
    end
    for o = 1:numOut
        output(o) = computeActivation(hiddenOutput',outputWeights(1:numHidden,o),outputWeights(outputBiasIndex,o));
        results(p+1,:,o) = [round(output(o)),round(testAns(o,p))];
    end
    % Determine parameters for Precision & Recall & write output to EXCEL
    for o = 1:numOut
        binaryOutput = round(output(o));
        binaryTestAns = round(testAns(o,p));
        %Confusion matrix parameters
        if(binaryOutput)
            confOutputs(p) = o;
        end
        if(binaryTestAns)
            confTargets(p) = o;
        end
        if(binaryOutput == binaryTestAns)
            if(binaryOutput == 1)
                truePositives(o) = truePositives(o) + 1;
            elseif(binaryOutput ==0)
                trueNegatives(o) = trueNegatives(o) + 1;
            end
        elseif(binaryOutput ~= binaryTestAns)
            if(binaryOutput == 1)
                falsePositives(o) = falsePositives(o) + 1;
            elseif(binaryOutput == 0)
                falseNegatives(o) = falseNegatives(o) + 1;
            end
        end
    end
end
disp('Done.');
% Map any non-classifieds as NaN
for i = 1:numTestPatterns
    if(confOutputs(i) == 0)
       confOutputs(i) = NaN;
    end
    if(confTargets(i) == 0)
        confTargets(i) = NaN;
    end
end
% Calculate Precision & Recall
precision = truePositives./(truePositives+falsePositives);
recall = truePositives./(truePositives+falseNegatives);

% Generate confusion matrix
C = confusionmat(confTargets,confOutputs);
%% Section 3: Write to Excel Spreadsheet

filename = 'results.xlsx';
disp('Writing to Excel file...');

% Write results
for o = 1:numOut
    sheet = strcat('Class ',int2str(o),' Results');  
    xlswrite(filename,results(:,:,o),sheet);
    xlswrite(filename,[cellstr('Predicted'),cellstr('Actual')],sheet);
end

% Write confusion matrix & precision/recall results
sheet = 'PrecRecConf Results';
xlswrite(filename,[cellstr('PRECISION'),cellstr('RECALL')],sheet,'B1:C1');
xlswrite(filename,[cellstr('Class 1');cellstr('Class 2');cellstr('Class 3')],sheet,'A2:A4');
xlswrite(filename,precision',sheet,'B2:B4');
xlswrite(filename,recall',sheet,'C2:C4');
xlswrite(filename,C,sheet,'C9:E11');

% Write final weight matricies
sheet = 'Hidden-to-Output Weights';
xlswrite(filename,outputWeights,sheet,'C3:E9');

sheet = 'Input-to-Hidden Weights';
xlswrite(filename,hiddenWeights,sheet,'C3:H14');

disp('Done.');