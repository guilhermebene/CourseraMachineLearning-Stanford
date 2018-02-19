function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Adding additional column to X
X = [ones(size(X, 1),1), X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a2 = sigmoid(X*Theta1'); % a2 is the hidden layer
a2 = [ones(m,1), a2];
a3 = sigmoid(a2*Theta2'); %a3 is the output layer

% a3 will be a (5000 x 10) matrix_type
% We need to find the highest value for each of the 5000 prediction examples
% Thus we need to choose for each of the 5000 lines, which of the 10 labels is
% the most appropriate. In other words, we need to find max value for each line.
[value, index] = max(a3, [],2);
p = index;
% =========================================================================
end
