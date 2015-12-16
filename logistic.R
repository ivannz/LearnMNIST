#### An R script implementing the Logistic Regression pipeline.

## Logistic regression
## $$ p_k(x) = \frac{ e^{f_k(x)} }{ \sum_{j=1}^L e^{f_j(x)} } \,, $$
##    where $f_k(x) = x' W_k $ and $x$ is augmented with extra 1 if bias
##    term is required.
## However softmax is invariant under uniform shifts of $(f_k)_{k=1}^L$. This makes
## the coefficients poorely identified. Therefore we have to impose a constraint
## that $(f_k)$ sum to $0$.
#### TODO!!!

## l( x, y ) = - \sum_{l=1}^L e_{yl} \log p_l(x)
## L( X, y ) = n^{-1} \sum_{i=1}^n l( x_i, y_i )

## Straighforward differentiation yields (dependency on $x$ is omitted)
##   $$ \frac{\partial p_k}{\partial f_k}
##          = \frac{ e^{f_k} }{ \sum_j e^{f_j} }
##          - \frac{ e^{f_k} e^{f_k} }{ (\sum_j e^{f_j} )^2}
##          = p_k (1 - p_k) \,,$$
## and for $l \neq k$
##   $$ \frac{\partial p_l}{\partial f_k}
##          = - \frac{ e^{f_l} e^{f_k} }{ (\sum_j e^{f_j} )^2}
##          = - p_l p_k \,. $$
## Thus
##   $$ \frac{\partial l}{\partial f_k}
##          = - \sum_{m=1}^L \frac{ e_{ym} }{p_l(x)}\frac{\partial p_m}{\partial f_k}
##          = - \frac{ e_{yk} }{p_k(x)} p_k(x) (1 - p_k(x))
##              - \sum_{m\neq k} -\frac{ e_{ym} }{p_m(x)} p_m(x) p_k(x)
##          = - e_{yk} + p_k(x) \,. $$
## Now
##   $$ \frac{\partial f_k}{\partial W_l}
##          = \begin{cases} x & \text{ if } l = k\\ 0 \text{ o/w } \end{cases} \,. $$
## Therefore the gradient of $l$ at $(x, y)$ with respect to
## the parameters are given by
##   $$ \frac{\partial l}{\partial W_k}
##          = \sum_{m=1}^L \frac{\partial l}{\partial f_m} \frac{\partial f_m}{\partial W_k} 
##          = \frac{\partial l}{\partial f_k} \frac{\partial f_k}{\partial W_k} 
##          = x ( p_k(x) - e_{yk} ) \,, $$
##  where $x\in \mathbb{R}^{d\times 1}$. Finally, the gradient of $ L $ is
##   $$ \frac{\partial L}{\partial W_k}
##          = \sum_{i=1}^n x_i ( p_k(x_i) - e_{y_ik} ) \,. $$
## Now a regularize term is usually additive, whence its gradient is just added
##  to the loss gradient:
##   $$ \nabla L + \nabla \Omega \,. $$
##  I use $L^2$ regularizer (bias terms are not regularized). Here it is 
##   $$ \Omega( W )
##          = \frac{\lambda}{2} tr( W' W ) 
##          = \frac{\lambda}{2} \sum_k W_k' W_k\,, $$
##  the derivative of which is
##   $$ \frac{\partial \Omega }{\partial W_k} = \lambda W_k \,. $$

## in R matrix-vector elementwise operation are done over columns
## IE R unravels matrix in column order and then applies the operation
##  recylcing the original vector, if needed.

#### PRIVATE FUNCTIONS
.get_update <- function( momentum = c( "simple", "nesterov", "none" ),
                         beta = 0, rms_weight = 0 ) {
## Creates a special function, which accepts the gradient, the current
##  estimate of the coefficients and the learning_rate, and does a
##  gradient descent step.
## Gradient adjustment step
    if( rms_weight > 0 ) {
## RMS prop attempts to approximate the diagonal of the Hessian
##  through exponentially weighted $\nabla f(\theta_t) \nabla f'(\theta_t)$.
##  Alternatively, this estimates the accumulated EWMA variance
##  of each element of the gradient. The higher the variance,
##  the more iiregular the direction is, hence the more tamed
##  should its movements be.
        acc_grad_sq <- 1.0
        grad_adjust_fn <- function( grad ) {
                acc_grad_sq <<- ( 1 - rms_weight ) * ( grad ** 2 ) + rms_weight * acc_grad_sq
                grad / sqrt( acc_grad_sq )
            }
    } else {
        grad_adjust_fn <- identity
    }
    
## SGD update functions
## c.f. Sutskever, Martens et al. 2013, Proceedings of the 30-th
##      International Conference on Machine Learning, Atlanta, Georgia, USA, 2013.
    if( momentum == "nesterov" ) {
        if( beta <= 0 || beta >= 1 )
            stop( "To use SGD with momentum you must specify a decay rate 'beta' within (0,1)" )
## Return an update step with Nesterov momentum
        acc_grad_ <- 0
        update_fn <- function( theta, learning_rate, grad ) {
                step_ <- beta * acc_grad_
                acc_grad_ <<- step_ + learning_rate * grad_adjust_fn( grad( theta - step_ ) )
                return( theta - acc_grad_ )
            }
    } else if( momentum == "simple" && beta > 0 ) {
        if( beta <= 0 || beta >= 1 )
            stop( "To use SGD with momentum you must specify a decay rate 'beta' within (0,1)" )
## Return an update step with simple (Classical) momentum
        acc_grad_ <- 0
        update_fn <- function( theta, learning_rate, grad ) {
                acc_grad_ <<- learning_rate * grad_adjust_fn( grad( theta ) ) + beta * acc_grad_
                return( theta - acc_grad_ )
            }
    } else if( momentum == "none" ) {
## Return an SGD update step
        update_fn <- function( theta, learning_rate, grad ) {
                return( theta - learning_rate * grad_adjust_fn( grad( theta ) ) )
            }
    } else
        stop( "Supported momementum modes are: 'none', 'simple' and 'nesterov'" )

    return( update_fn )
## Use environment( updater_ ) to inspect the variables in
##   the functional closure.
}

##### R's matrices are columnwise.

## Creates a round-robin group assignment for an array of length n
.group <- function( n, m )
    rep( 1 : ( ( n + m - 1 ) %/% m ), m )[ 1 : n ]

## Computes the softmax prediction of the logistic regression
.predict_proba <- function( theta, X, .log = FALSE ) {
## Compute the probabilities inferred by the model for
##   the given input samples X.
##  theta[matrix] -- the coefficients of the logisitc regression;
##  X[matrix] -- the current coefficients of the log regression model;
##  .log[bool] -- If this is set to TRUE, the logarithms of
##        probabilities are returned.

## Compute the terms in exponents
    f_ <- X %*% theta
## Normalise the exponents
    f_star <- f_ - apply( f_, 1, max )
    p_star <- exp( f_star )
## Compute the sum-exp
    sum_p_star <- apply( p_star, 1, sum )
## return Softmax (or its logarithm)
    return( if( !.log ) {
            p_star / sum_p_star
        } else {
            f_star - log( sum_p_star )
        } )
}

.gradient <- function( theta, X, y ) {
## Compute the gradient over provided dataset (X, y).X (2D) and y (1D) at theta (1D)
##  theta[matrix] -- the coefficients of the logisitc regression;
##  X[matrix] -- the input samples;
##  y[vector] -- target values (class labels).
    p <- .predict_proba( theta, X )
## The gradient is given \nabla L = by n^{-1} \sum_{i=1}^n x_i ( p_k(x_i) - e_{y_ik} ) :
##   equivalent (but slighty faster than t( X ) %*% loss ).
    crossprod( X, p - diag( dim( theta )[ 2 ] )[ y, ] ) / nrow( X )
}

.regularize_theta <- function( lambda, theta, grad, intercept = 0 ) {
## Apply L^2 regularizer to the given gradient at the current
##  estimates of coefficients.
##   lambda[numeric] -- the weight of the L^2 regualrizer.
##   theta[matrix] -- the coefficients of the logisitc regression;
##   grad[matrix] -- object of the same shape as theta;
##   intercept[integer] -- the row number of the unit columns, most usually 1).
## Replicate the L^2 regularizer weights
    lambda_ <- rep( lambda, dim( theta )[ 1 ] )
## the first row is bais weights, which is never regularized.
    if( intercept > 0 )
        lambda_[ 1 ] <- 0
## regularize
    grad + lambda_ * theta
}

## Compute the logloss
.logloss <- function( X, y, theta ) {
    proba_ <- .predict_proba( theta, X )
    proba_ <- proba_[ matrix( c( 1 : nrow( X ), y ), ncol = 2 ) ]
    - mean( log( ifelse( proba_ > 0.0, ifelse( proba_ < 1.0, proba_, 1 - 1e-14 ), 1e-14 ) ) )
}

#### PUBLIC INTERFACE
## Returns an environment (a container) for the model
init <- function( add_intercept = TRUE, lambda = 1.0, standardize = TRUE )
## Create an instance of the logisitc regression model.
##   add_intercept[boolean] -- whether to add a bias term to
##         the logisitc regression;
##   lambda[numeirc] -- the weight of L^2 regularizer;
##   standardize[boolean] -- determines whether the data should
##         be scaled to unit variance as well as mean shifted.
    list2env( list( add_intercept = add_intercept,
                    standardize = standardize,
                    lambda = lambda ),
              parent = emptyenv( ) )

fit <- function( model, X, y,
                 niter = 1000, learning_rate = 0.01,
                 momentum = "nesterov", beta = 0.9,
                 rms_weight = 0, batch_size = 32,
                 verbose = 0, ... ) {
## Estimates a logisitc regression model using the specified
##   tecnique over the training dataset (X, y).
##  model -- a fitted instance of the logistic regression model;
##  X[matrix] -- the input samples;
##  y[vector] -- target values (class labels);
##  niter[integer] -- the nuber of epochs of learning (iterations);
##  learning_rate[numeric, function] -- the learning rate schedule
##        to apply. In 'numeric', then the schedule is flat. Otherwise,
##        the learning rate is received from the function(kiter, niter)
##        where kiter is the current iteration, and niter is the total
##        number of iterations;
##  momentum[character] -- Selects which momentun mode to use: 
##        implemented modes are 'none', 'simple' and 'nesterov';
##  beta[double] -- the momentum paramter, determines the inertia of gradinets;
##  rms_weight[double] -- the EWMA-like parameter of the RMS-prop (Hinton, 2012)
##        technique, used to account for approximate curvature of
##        the loss;
##  batch_size[integer] -- the size of batch stochastic gradient descent;
##  verbose[integer] -- determines the volume of debug and service messages
##        printed.

## __docstring__
    stopifnot( is.environment( model ) )
    model$theta_ <- NULL
## Get the classes
    model$classes_ <- sort( c( unique( y ) ) )
## Define the dimensions of the weigh matrix
    model$dim_ <- c( ncol( X ) + if( model$add_intercept ) 1 else 0,
                     length( model$classes_ ) )
    intercept_ <- if( model$add_intercept ) 1 else 0
## Learn data standartization coefficients
    mean_ <- apply( X, 2, mean )
    std_ <- rep( 1.0, length( mean_ ) )
    if( model$standardize ) {
        std_ <- apply( X, 2, sd )
        std_[ std_ == 0 ] <- 1.0
    }
## Map labels into internal classes
    labels_ <- match( y, model$classes_ )
## This grouping schedule need not change every iteration, since
##  uniformly random permutations are to be grouped with it.
    group_ <- .group( nrow( X ), batch_size )
## Get the required updater
    updater_ <- .get_update( momentum = momentum, beta = beta, rms_weight = rms_weight )
## use environment( updater_ ) to inspect its closure.
## Setup the learning rate schedule
    if( !is.function( learning_rate ) )
## R evaluates lazily, so create and expression, parse and then compile it
        learning_rate <- eval( parse( text = sprintf( "function( kiter ) %g",
                                                      learning_rate ) ) )
## Do the batch SGD loop
    kiter <- 1
## Intialize $\theta$
    theta_ <- matrix( runif( model$dim_[ 2 ] * model$dim_[ 1 ] ) - 0.5,
                      nrow = model$dim_[ 1 ] )
    tryCatch( { repeat {
## Stop if the number of iteration has been exceeded.
        if( kiter > niter ) break
## Get the current learning rate
        learning_rate_ <- learning_rate( kiter )
## Assign observations to groups at random
        batches_ <- tapply( sample.int( nrow( X ), replace = FALSE ),
                            group_, c, simplify = TRUE )
## Iterate over batches
        epoch_loss_ <- 0
        for( batch_group_ in names( batches_ ) ) {
            batch_ <- batches_[[ batch_group_ ]]
## Get the current batch : normalise the data here -- less memory pressure, more cpu though
            X_ <- t( ( t( X[ batch_, ] ) - mean_ ) / std_ )
            y_ <- labels_[ batch_ ]
## Add the intercept term.
            if( intercept_ > 0 ) X_ <- cbind( 1, X_ )
## Create a gradient by currinyg the batch function
            grad_ <- function( theta ) {
                    grad_ <- .gradient( theta, X_, y_ )
                    .regularize_theta( model$lambda, theta, grad_, intercept = intercept_ )
            }
## Do the update
            theta_new <- updater_( theta = theta_,
                                   learning_rate = learning_rate_,
                                   grad = grad_ )
## Compute the new loss
            if( verbose > 0 ) {
                batch_loss_ <- .logloss( X_, y_, theta_new )
                epoch_loss_ <- epoch_loss_ + batch_loss_ * batch_size
            }
## DEBUG report
            if( verbose > 10 ) {
                norm_ <- sqrt( sum( ( theta_new - theta_ ) ** 2 ) )
                cat( sprintf( "%s l^2 norm %f, loss %f\n",
                              batch_group_, norm_, batch_loss_ ) )
            }
## Commit the update
            theta_ <- theta_new
        }
## DEBUG report
        if( verbose > 0 ) {
            norm_ <- sqrt( sum( ( theta_new - theta_ ) ** 2 ) )
            cat( sprintf( "epoch %d, mean loss %f\n",
                          kiter, epoch_loss_ / nrow( X ) ) )
        }

        kiter <- kiter + 1
    } }, interrupt = function( e ) NULL,
         condition = signalCondition )
## Update the model instance
    model$intercept_ <- intercept_
    model$mean_ <- mean_
    model$std_ <- std_
    model$theta_ <- theta_
    return( model )
}

predict_proba <- function( model, X ) {
## Compute the probabilities inferred by the model for
##   the given input samples X.
##  model -- instance of a fitted logistic regression model;
##  X[matrix] -- the input samples.

    stopifnot( !is.null( model$theta_ ) )
    stopifnot( dim( X )[ 2 ] == model$dim_[ 1 ] - model$intercept_ )
## Standardize the data
    X <- t( ( t( X ) - model$mean_ ) / model$std_ )
## Add the intercept if necessary
    if( model$intercept_ > 0 ) X <- cbind( 1, X )
## Predict the class probabilities
    .predict_proba( model$theta_, X, .log = FALSE )
}

predict <- function( model, X ) {
## Compute the classes predicted by the model for
##   the given input samples X.
##  model -- instance of a fitted logistic regression model;
##  X[matrix] -- the input samples.

    proba_ <- predict_proba( model, X )
## Predict the label using MAP rule
    model$classes_[ apply( proba_, 1, which.max ) ]
}

logloss <- function( model, X, y ) {
## Compute the loss of logistic regresssion (multinomial logloss)
##   over some dataset (X, y).
##  model -- a fitted instance of the logistic regression model;
##  X[matrix] -- the input samples;
##  y[vector] -- target values (class labels).

## Predict the class probabilities
    proba_ <- predict_proba( model, X )
## encode the labels
    labels_ <- match( y, model$classes_ )
## get the predicted probability of the actual calss
    proba_ <- proba_[ matrix( c( 1 : nrow( X ), labels_ ), ncol = 2 ) ]
## Compute truncated mean log loss
    - mean( log( ifelse( proba_ > 0.0, ifelse( proba_ < 1.0, proba_, 1 - 1e-14 ), 1e-14 ) ) )
}

#### User accessible functions
learnModel <- function( data, labels ) {
    model <- init( add_intercept = TRUE, lambda = 0.1, standardize = TRUE )
    fit( model, data, labels, 
         niter = 10, learning_rate = 0.01,
         momentum = "nesterov", beta = 0.8,
         rms_weight = 0, batch_size = 512,
         verbose = 1 )
}

testModel <- function( model, data ) predict( model, X = data )

### Sandbox
if( FALSE ) {
    classifier <- learnModel( data = trainData, labels = trainLabels )
    classifier <- logloss( classifier, data = trainData, labels = trainLabels )

    images_ <- classifier$theta_[-1,]
    images_ <- ( images_ - min( images_ ) ) / ( max( images_ ) - min( images_ ) )

    for( n in 1:10 )
        image( t(matrix(images_[,n], ncol=28, nrow=28)), Rowv=28, Colv=28, col = heat.colors(256),  margins=c(5,10))

}

