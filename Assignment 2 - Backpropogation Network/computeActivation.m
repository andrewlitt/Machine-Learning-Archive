function a = computeActivation(inputs,weights,bias)
net = dot(inputs',weights') + bias;
a = 1/(1 + exp(-net));
end

