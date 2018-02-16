function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%=====COST=======
%Working with the vectorized form, so I had to subtract the regularization contribution of the theta(1) element, which cannot be regularized
%The vectorized form considers the entire vector theta, so this subtraction is necessary
J = (1/m)*sum(-y' .* log(sigmoid((theta')*X')) - (1.-y)'.*log(1.-sigmoid((theta')*X'))) + (lambda/2/m)*sum(theta.^2) - (lambda/2/m)*theta(1)^2;

%=====GRADIENT======
%Again, the theta(1) element does not suffer regularization, so its resstrict expression is described below
grad(1) = (1/m)*sum((sigmoid((theta')*X') - y')*X(:,1));
%For all other theta´s elements, regularization is applied via vectorizing as described below
for j=2:size(theta)(1)
  grad(j) = (1/m)*sum((sigmoid((theta')*X') - y')*X(:,j)) + (lambda/m)*theta(j);
end
% =============================================================

end
