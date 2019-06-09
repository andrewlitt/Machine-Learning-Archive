% CMPE 452 Assignment 1 - Implementation of a Perceptron 
% PerceptronTest Function
% Andrew Litt
% 10150478, 14asl
function [w,iter] = PerceptronTrain(inputs, targets, wi, c, epochs)
% inputs - testing dataset
% targets - testing answer set
% wi - initial weight vector
% c - learning rate
% epochs - maximum iterations

[l,s] = size(inputs);
w = wi;

%Initialize iterations and error occurance counters
iter = 0;
errors = -1;

% loop until max iterations or until perceptron converges
while (iter < epochs) & (errors ~= 0)
    errors = 0;
    for i = 1:s                       % for each sample data vector
        x = inputs(:,i);                % get the data points
        y = dot(x,w(1:l))+ w(l+1);      % multiply them by the weights
        y = heaviside(y);               % cast to binary output with step function
        d = targets(i);                 % get the desired output
        if d ~= y                       % if the output doesn't match
            er = d - y;                     % determine whether to +/-
            w = w + [er*c*x; er*c*w(l+1);]; % adjust the weights
            errors = errors + 1;             % acknowledge the error to iterate again
        end
    end
    iter = iter + 1;                    % add the iteration count
end

end

