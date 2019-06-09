% CMPE 452 Assignment 1 - Implementation of a Perceptron 
% PerceptronTest Function
% Andrew Litt
% 10150478, 14asl
function [y,errors] = PerceptronTest(x,d,w)
[l,s] = size(x); % get input dimensions: l=datapoints per sample, s=samples
errors = zeros(1,s); % initialize data points
y = zeros(1,s);

for i = 1:s                     % for each testing data point 
    xi = x(:,i);                    % retrieve the data
    yi = dot(xi,w(1:l)) + w(l+1);   % multiply them by the weights
    y(i) = heaviside(yi);           % cast to binary output with step function
    if d(i) ~= y(i)                 % if it doesn't match the desired output
        errors(1,i) = 1;                % add an error flag
    end
end

