function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1) X];

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

for i = 1:m
  % create expected output vector size k,1 where correct ouput is 1, others 0
  yk = zeros(num_labels, 1);
  yk(y(i)) = 1;

  % compute h(x)
  a1 = X(i,:);
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [1 a2];
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  %backprop
  d3 = a3 - yk';
  d2 = (d3 * Theta2)(2:end) .* sigmoidGradient(z2);
  Theta1_grad = Theta1_grad + (a1' * d2)';
  Theta2_grad = Theta2_grad + (a2' * d3)';

  for k = 1:num_labels
    step_cost = -yk(k) * log(a3(k)) - (1 - yk(k)) * log(1 - a3(k));
    J = J + step_cost;
  end
end
J = J / m;

Theta1_regularization = (lambda / m) * Theta1;
Theta1_regularization(:,1) = 0;
Theta2_regularization = (lambda / m) * Theta2;
Theta2_regularization(:,1) = 0;
Theta1_grad = Theta1_grad / m + Theta1_regularization;
Theta2_grad = Theta2_grad / m + Theta2_regularization;

% Regularization
theta1_without_bias = Theta1(:, 2:input_layer_size+1);
theta1_elementwise_squared = theta1_without_bias .^ 2;
regularization_first_term = sum(theta1_elementwise_squared(:));

theta2_without_bias = Theta2(:, 2:hidden_layer_size+1);
theta2_elementwise_squared = theta2_without_bias .^ 2;
regularization_second_term = sum(theta2_elementwise_squared(:));

regularization_term = (lambda / (2 * m)) * ...
  (regularization_first_term + regularization_second_term);

J = J + regularization_term;
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
