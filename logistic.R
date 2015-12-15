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

#### Internal functions
"get_update" <- function( momentum = c( "simple", "nesterov", "none" ), beta = 0 ) {
## c.f. Sutskever, Martens et al. 2013, Proceedings of the 30-th
##      International Conference on Machine Learning, Atlanta, Georgia, USA, 2013.
    if( momentum == "nesterov" ) {
        if( beta <= 0 || beta >= 1 )
            stop( "To use SGD with momentum you must specify a decay rate 'beta' within (0,1)" )
## Return an update step with Nesterov momentum
        acc_grad_ <- 0
        update_fn <- function( theta, learning_rate, grad ) {
                step_ <- beta * acc_grad_
                acc_grad_ <<- step_ + learning_rate * grad( theta - step_ )
                return( theta - acc_grad_ )
            }
    } else if( momentum == "simple" && beta > 0 ) {
        if( beta <= 0 || beta >= 1 )
            stop( "To use SGD with momentum you must specify a decay rate 'beta' within (0,1)" )
## Return an update step with simple (Classical) momentum
        acc_grad_ <- 0
        update_fn <- function( theta, learning_rate, grad ) {
                acc_grad_ <<- learning_rate * grad( theta ) + beta * acc_grad_
                return( theta - acc_grad_ )
            }
    } else if( momentum == "none" ) {
## Return an SGD update step
        update_fn <- function( theta, learning_rate, grad ) {
                return( theta - learning_rate * grad( theta ) )
            }
    } else
        stop( "Supported momementum modes are: 'none', 'simple' and 'nesterov'" )
    return( update_fn )
}

## R's matrices are columnwise, so this maintains the original ordering.

## Creates a round-robin group assignment for an array of length n
.group <- function( n, m )
    rep( 1 : ( ( n + m - 1 ) %/% m ), m )[ 1 : n ]

## Computes the softmax prediction of the logistic regression
.predict_proba <- function( X, theta, .log = FALSE ) {
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

## Compute the gradient over X (2D) and y (1D) at theta (1D)
.gradient <- function( X, y, theta ) {
## Softmax
    p <- .predict_proba( X, theta )
## The gradient is given \nabla L = by n^{-1} \sum_{i=1}^n x_i ( p_k(x_i) - e_{y_ik} ) :
##   equivalent (but slighty faster than t( X ) %*% loss ).
    crossprod( X, p - diag( dim( theta )[ 2 ] )[ y, ] ) / nrow( X )
}

.regularize_theta <- function( lambda, theta, grad, intercept = 0 ) {
## The L^2 regularizer
    lambda_ <- rep( lambda, dim( theta )[ 1 ] )
## the first row is bais weights, which is never regularized.
    if( intercept > 0 )
        lambda_[ 1 ] <- 0
## regularize
    grad + lambda_ * theta
}

## Compute teh log loss
.logloss <- function( X, y, theta ) {
    proba_ <- .predict_proba( X, theta )
    proba_ <- proba_[ matrix( c( 1 : nrow( X ), y ), ncol = 2 ) ]
    - sum( log( ifelse( proba_ > 0.0, ifelse( proba_ < 1.0, proba_, 1 - 1e-14 ), 1e-14 ) ) ) / nrow( X )
}

## Returns an environment (a container) for the model
init <- function( add_intercept = TRUE, lambda = 1.0, standardize = TRUE )
    list2env( list( add_intercept = add_intercept,
                    standardize = standardize,
                    lambda = lambda ),
              parent = emptyenv( ) )

fit <- function( model, X, y,
                 niter = 1000, learning_rate = 0.01,
                 momentum = "nesterov", beta = 0.9,
                 batch_size = 32, verbose = 0, ... ) {
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
    updater_ <- get_update( momentum = momentum, beta = beta )
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
                    grad_ <- .gradient( X_, y_, theta )
                    .regularize_theta( model$lambda, theta, grad_, intercept = intercept_ )
            }
## Do the update
            theta_new <- updater_( theta = theta_,
                                   learning_rate = learning_rate_,
                                   grad = grad_ )
## Compute the new loss
            batch_loss_ <- .logloss( X_, y_, theta_new )
            if( verbose > 10 ) {
                norm_ <- sqrt( sum( ( theta_new - theta_ ) ** 2 ) )
                cat( sprintf( "%s l^2 norm %f, loss %f\n",
                              batch_group_, norm_, batch_loss_ ) )
            }
            theta_ <- theta_new
            epoch_loss_ <- epoch_loss_ + batch_loss_ * batch_size
        }
        if( verbose > 0 ) {
            norm_ <- sqrt( sum( ( theta_new - theta_ ) ** 2 ) )
            cat( sprintf( "epoch %d, mean loss %f\n",
                          kiter, epoch_loss_ / nrow( X ) ) )
        }
        kiter <- kiter + 1
    } }, interrupt = function( e ) NULL,
         condition = signalCondition )
## Return a model instance
    model$intercept_ <- intercept_
    model$mean_ <- mean_
    model$std_ <- std_
    model$theta_ <- theta_
    return( model )
}

## Predict the class probability for the given data
predict_proba <- function( model, data ) {
    stopifnot( !is.null( model$theta_ ) )
    stopifnot( dim( data )[ 2 ] == model$dim_[ 1 ] - model$intercept_ )
## Standardize the data
    data_ <- t( ( t( data ) - model$mean_ ) / model$std_ )
## Add the intercept if necessary
    if( model$intercept_ > 0 ) data_ <- cbind( 1, data_ )
## Predict the class probabilities
    .predict_proba( data_, model$theta_, .log = FALSE )
}

predict <- function( model, data ) {
## Compute the inferred probabilities
    proba_ <- predict_proba( model, data )
## Predict the label using MAP rule
    model$classes_[ apply( proba_, 1, which.max ) ]
}

#### User accessible functions
learnModel <- function( data, labels ) {
    model <- init( add_intercept = TRUE, lambda = 0.1, standardize = TRUE )
    fit( model, data, labels, 
         niter = 10, learning_rate = 0.01,
         momentum = "nesterov", beta = 0.8,
         batch_size = 512, verbose = 1 )
}

testModel <- function( model, data ) predict( model, data )

### Sandbox
if( FALSE ) {
    classifier <- learnModel(data = trainData, labels = trainLabels)

    images_ <- classifier$theta_[-1,]
    images_ <- ( images_ - min( images_ ) ) / ( max( images_ ) - min( images_ ) )

    for( n in 1:10 )
        image( t(matrix(images_[,n], ncol=28, nrow=28)), Rowv=28, Colv=28, col = heat.colors(256),  margins=c(5,10))




    ####### Func Test
    theta <- matrix( 1:12, 3, 4 )
    grad <- function( theta )
        return( do.call( matrix, args = c( list( 1 ), dim(theta) ) ) )

    updater_ <- get_update(momentum = "nesterov", beta = 0.9)

    for (i in 1:10) {
        theta <- updater_( theta, 0.5, grad )
        print( theta )
    }
    environment( updater_ )$acc_grad_

    theta <- runif( 4*11 )
    sum(abs(do.call( .pack, args = .unpack(theta, nrow = 4))-theta))

}