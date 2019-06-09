%% CMPE 452 Assignment 3: PCA Network
% Andrew Litt
% 10150478, 14asl

clc
clear all
% Prints system parameters to results.txt
% Saves output to both output.csv and output.wav

%% Section 1: Parse Dataset & Set up System Parameters

data = xlsread("sound.csv");

c = 1; %Learning Rate

range = 1; % Range of possible random weight initialization
W = -range + 2*range*rand(1,2); %Initialize random weights from -range<->+range
initialWeights = W;

%% Section 2: Calculate output 

epochs = 1;

%For however many epochs it takes to effectively train
for k = 1:epochs
    %Iterate through each data sample
    for i = 1:length(data)
        x = [data(i,1), data(i,2)];     % Get the data sample
        y = dot(x,W);                   % Multiply by weights
        K = y*y;                        % K option as suggested by Williams
        deltaW = c*y*x - K*W;           % Calculate weight change
        W = W + deltaW;                 % Change weights
    end
end 

%Write to output file with trained weights.
output = [];
for j = 1:length(data)
    x = [data(j,1), data(j,2)];     % Get data sample
    output = [output; dot(x,W)];    % Append new output to output file 
end

%% Section 3: Normalize & Write Output

% Normalize output datafile
maxFeature = 0;
minFeature = inf;
for k = 1:length(output)
    if(output(k) > maxFeature)
        maxFeature = output(k);
    end
    if(output(k) < minFeature)
        minFeature = output(k);
    end
end
for k = 1:length(output)
    output(k) = (output(k)-minFeature)/(maxFeature-minFeature);
end

% Write data to audio file and output spreadsheet
audiowrite('output.wav',output,8000);
xlswrite('output.csv',output);

% Output learning rate, initial and final weight vectors
fileID = fopen('results.txt','w');
fprintf(fileID,'Learning Rate:  %.2f \r\n', c);
fprintf(fileID,'Initial Weights: %.6f %.6f\r\n', initialWeights);
fprintf(fileID,'Final Weights:   %.6f %.6f\r\n', W);
fclose(fileID);


