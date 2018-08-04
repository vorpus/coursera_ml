function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=1:m
  predict = sigmoid(X(i,:)*theta);
  J = J - y(i)*log(predict) - (1 - y(i))*log(1-predict);
end
regular = 0;
for j=2:n
  regular = regular + theta(j)^2;
end

J = J/m + regular*lambda/(2*m);

for i=1:m
  predict = sigmoid(X(i,:)*theta);
  grad(1) = grad(1) + (predict - y(i)) * X(i, 1);
  for j=2:n
    grad(j) = grad(j) + (predict - y(i)) * X(i, j);
  end
end

grad(1) = grad(1) / m;
for j=2:n
  grad(j) = (grad(j)/m) + (lambda * theta(j))/m;
end

% =============================================================

end
