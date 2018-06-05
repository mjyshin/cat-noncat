using Images
using PyPlot
include("load_data.jl")


function sigmoid(z)
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1./(1 + exp.(-z))
    ### END CODE HERE ###

    return s
end


function initialize_with_zeros(dim)
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    w = zeros(dim,1)
    b = 0
    ### END CODE HERE ###

    assert(size(w) == (dim, 1))
    assert(isa(b, Float64) || isa(b, Int64))

    return w, b
end


function propagate(w, b, X, Y)
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (number of examples, 1)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = size(X,1)

    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(X*w + b)                                    # compute activation
    cost = -1/m*sum(Y.*log.(A) + (1 - Y).*log.(1 - A))      # compute cost
    ### END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = 1/m*X'*(A - Y)
    db = 1/m*sum(A - Y)
    ### END CODE HERE ###

    assert(size(dw) == size(w))
    assert(typeof(db) == Float64)
    assert(size(cost) == ())

    grads = Dict("dw" => dw,
                 "db" => db)

    return grads, cost
end


function optimize(w, b, X, Y, num_iterations, learning_rate; print_cost=false)
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (number of examples, 1)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []; dw = []; db = []

    for i = 1:num_iterations

        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ###
        grads, cost = propagate(w,b,X,Y)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate*dw
        b = b - learning_rate*db
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0
            push!(costs,cost)
        end

        # Print the cost every 100 training examples
        if print_cost && i % 100 == 0
            println(@sprintf "Cost after iteration %i: %f" i cost)
        end
    end

    params = Dict("w" => w,
                  "b" => b)

    grads = Dict("dw" => dw,
                 "db" => db)

    return params, grads, costs
end


function predict(w, b, X)
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (number of examples, num_px * num_px * 3)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    m = size(X,1)
    Y_prediction = zeros(Int,m,1)
    w = reshape(w, size(X,2), 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(X*w + b)

    ### END CODE HERE ###

    for i = 1:size(A,1)

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        A[i] > 0.5 ? Y_prediction[i] = 1 : Y_prediction[i] = 0
        ### END CODE HERE ###
    end

    assert(size(Y_prediction) == (m, 1))

    return Y_prediction
end


function model(X_train, Y_train, X_test, Y_test; num_iterations=2000, learning_rate=0.5, print_cost=false)
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    ### START CODE HERE ###

    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(size(X_train,2))

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)

    ### END CODE HERE ###

    # Print train/test Errors
    println("train accuracy: $(100 - mean(abs.(Y_prediction_train - Y_train)) * 100) %")
    println("test accuracy: $(100 - mean(abs.(Y_prediction_test - Y_test)) * 100) %")


    d = Dict("costs" => costs,
             "Y_prediction_test" => Y_prediction_test,
             "Y_prediction_train" => Y_prediction_train,
             "w" => w,
             "b" => b,
             "learning_rate" => learning_rate,
             "num_iterations" => num_iterations)

    return d
end

function main()
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()

    ### START CODE HERE ### (≈ 3 lines of code)
    m_train = size(train_set_x_orig,1)
    m_test = size(test_set_x_orig,1)
    num_px = size(train_set_x_orig,2)
    ### END CODE HERE ###

    println("Number of training examples: m_train = ", m_train)
    println("Number of testing examples: m_test = ", m_test)
    println("Height/Width of each image: num_px = ", num_px)
    println("Each image is of size: (", num_px, ", ", num_px, ", 3)")
    println("train_set_x shape: ", size(train_set_x_orig))
    println("train_set_y shape: ", size(train_set_y))
    println("test_set_x shape: ", size(test_set_x_orig))
    println("test_set_y shape: ", size(test_set_y))

    # Reshape the training and test examples

    ### START CODE HERE ### (≈ 2 lines of code)
    train_set_x_flatten = reshape(train_set_x_orig, size(train_set_x_orig,1), :)
    test_set_x_flatten = reshape(test_set_x_orig, size(test_set_x_orig,1), :)
    ### END CODE HERE ###

    println("train_set_x_flatten shape: ", size(train_set_x_flatten))
    println("train_set_y shape: ", size(train_set_y))
    println("test_set_x_flatten shape: ", size(test_set_x_flatten))
    println("test_set_y shape: ", size(test_set_y))
    println("sanity check after reshaping: ", train_set_x_flatten[1,1:5])

    train_set_x = train_set_x_flatten/255
    test_set_x = test_set_x_flatten/255

    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=true)

    # Example of a picture that was wrongly classified.
    index = 11
    imshow(reshape(test_set_x[index,:,:,:], (num_px, num_px, 3)))
    println("y = ", test_set_y[index], ", you predicted that it is a \"", classes[d["Y_prediction_test"][index]+1], "\" picture.")

    # Plot learning curve (with costs)
    costs = d["costs"]
    plot(costs)
    ylabel("cost")
    xlabel("iterations (per hundreds)")
    title("Learning rate = $(d["learning_rate"])")

    learning_rates = [0.01, 0.001, 0.0001]
    models = Dict()
    for i in learning_rates
        println("learning rate is: ", i)
        models[string(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, 1500, i, false)
        println("\n", "-------------------------------------------------------", "\n")
    end

    fig, ax = subplots()
    for i in learning_rates
        ax[:plot](models[string(i)]["costs"], label=models[string(i)]["learning_rate"])
    end

    ylabel("cost")
    xlabel("iterations")

    ax[:legend](loc="upper center", shadow=true, frameon=true, facecolor="0.90", edgecolor="black")
end


main()
