#### Derived matrics
.confusion2x2 <- function( label, table ) {
## Takes a precomputed confuson NxN matrix and produces
##   a 2x2 contingency table for the specified class in 
##   one-versus-all classification setting.
##  label[character] -- label of the class considered positive;
##  table[matrix] -- the source confusion matrix
    stopifnot( dimnames( table )$true == dimnames( table )$predicted )
## Check if the requested class is present
    classes_ <- dimnames( table )$true
    stopifnot( label %in% classes_ )
## Other classes except the positive
    rest <- classes_[ !( classes_ %in% label ) ]
## The table contents in column-first order (FORTRAN and R)
    table <- c( table[ label, label ],       sum( table[ rest, label ] ),
                sum( table[ label, rest ] ), sum( table[ rest, rest ] ) )
## Add propert dimnames
    matrix( table, ncol = 2, dimnames = list( predicted = c( "+", "-" ),
                                              true = c( "+", "-" ) ) )
  }

.precision <- function( confusion_cube )
## Computes the precision, which is TP / ( TP + FP )
  apply( confusion_cube, 3, function( T ) T["+","+"] / ( T["+","+"] + T["+","-"] ) )

.recall <- function( confusion_cube )
## Computes the recall, which is TP / ( TP + TN )
  apply( confusion_cube, 3, function( T ) T["+","+"] / ( T["+","+"] + T["-","+"] ) )

.F_measure <- function( confusion_cube ) {
## Compute the F-score : the harmonic average of precision and recall
    prc <- .precision( confusion_cube )
    rcl <- .recall( confusion_cube )
    2 * ( 1/prc + 1/rcl )
  }

.specificity <- function( confusion_cube )
## Specificity is the ratio of TN to ( TN + FP )
  apply( confusion_cube, 3, function( T ) T["-","-"] / ( T["-","-"] + T["+","-"] ) )

.fdr <- function( confusion_cube )
## Computes the Flase Discovery Rate, which is just one
##  minus precision.
  1 - .precision( confusion_cube )

## PUBLIC INTERFACE WRAPPER
.wrapper <- function( FUN, name = NULL ) {
    if( is.null( name ) )
      name <- gsub( "\\W", "", as.character( substitute( FUN ) ) )
## A handy wrapper fo making publicly available metrics
    return( function( true, predicted ) {
## Compute the reqired performance metric. Multiclass
##   labels are handled correctly.
##  true[vector] -- true class labels;
##  predicted[vector] -- predicted class labels.
      matrix_ <- confusion( true, predicted )
      result_ <- FUN( simplify2array(
              sapply( dimnames( matrix_ )$true, .confusion2x2,
                      matrix_, simplify = FALSE ) ) )
      matrix( result_, ncol = 1,
              dimnames = list( class = names( result_ ), name ) )
    } )
  }

## PUBLIC INTERFACE
## Compute precision, recall, F, specificity and FDR.
##  true[vector] -- true class labels;
##  predicted[vector] -- predicted class labels.
## See ".wrapper" for details
precision <- .wrapper( .precision )
recall <- .wrapper( .recall )
F_measure <- .wrapper( .F_measure )
specificity <- .wrapper( .specificity )
fdr <- .wrapper( .fdr )

confusion <- function( true, predicted ) {
## Compute the confusion matrix for the predicted class labels.
##  true[vector] -- true class labels;
##  predicted[vector] -- predicted class labels.
## Use tapply to map-reduce by (true, predicted) pairs
    matrix_ <- tapply( seq_along( true ), list( predicted, true ), length )
## Pairs that were never seen get a missing value
    matrix_[ is.na( matrix_ ) ] <- 0
## Add correct dimension names
    names( dimnames( matrix_ ) ) <- c( "predicted", "true" )
    matrix_
  }

roc_curve <- function( true, scores, .plot = TRUE ) {
## Compute ROC curves.
##  true[vector] -- true class labels;
##  scores[matrix] -- predicted score with proper column
##     names being the class lables. Must always be a matrix
##     with the number of columns equal to the number of
##     disticnt classes.
    stopifnot( length( dimnames( scores ) ) == 2 )
    roc <- sapply( dimnames( scores )[[ 2 ]], function( class_ ) {
## Rearrange the class's score in decreasing order
        decr_score <- order( scores[ , class_ ], decreasing = TRUE )
## Get the score and binarized labels in the correct order
        score_ <- scores[ decr_score, class_ ]
        true_ <- ( true[ decr_score ] == class_ )
## Compute the number of true/false positives with decreasing threshold
        list( fpr = cumsum( !true_ ) / sum( !true_ ),
              tpr = cumsum(  true_ ) / sum(  true_ ),
              thresholds = score_ )
      }, simplify = FALSE )
## Plot the curves if necessary
    if( .plot ) {
## Setup the plot
      plot( c(0,1), c(0,1), type = "l", col = "black", lwd = 0.5,
            lty = "dotted", main = "ROC curves", xlab = "fpr", ylab = "tpr",
            ask = TRUE )
## Pick colours
      colors_ <- structure( rainbow( length( roc ) ), names = names( roc ) )
## Plot all curves
      invisible( lapply( names( roc ),
                         function( class_ )
                            lines( roc[[ class_ ]]$fpr,
                                   roc[[ class_ ]]$tpr,
                                   col = colors_[ class_ ] ) ) )
## Add a legend
      legend( "bottomright", cex = 0.8,
              legend = sprintf( "Class %s", names( roc ) ),
              fill = colors_ )
    }
    invisible( roc )
  }
