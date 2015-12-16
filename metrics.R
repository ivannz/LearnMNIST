confusion <- function( true, test ) {
## Use tapply to map-reduce by (true,test) pairs
    matrix_ <- tapply( seq_along( true ), list( test, true ), length )
## Pairs that were never seen get a missing value 
    matrix_[ is.na( matrix_ ) ] <- 0
## Add correct dimension names
    names( dimnames( matrix_ ) ) <- c( "test", "true" )
    matrix_
}

confusion2x2 <- function( label, table ) {
## Takes a precomputed confuson NxN matrix and produces
##   a 2x2 contingency table for the specified class in 
##   one-versus-all classification setting.
##  label[character] -- label of the class considered positive;
##  table[matrix] -- the source confusion matrix
  stopifnot( dimnames( table )$true == dimnames( table )$test )
## Check if the requested class is present
  classes_ <- dimnames( table )$true
  stopifnot( label %in% classes_ )
## Other classes except the positive
  rest <- classes_[ !( classes_ %in% label ) ]
## The table contents in column-first order (FORTRAN and R)
  table <- c( table[ label, label ],       sum( table[ rest, label ] ),
              sum( table[ label, rest ] ), sum( table[ rest, rest ] ) )
## Add propert dimnames
  matrix( table, ncol = 2, dimnames = list( test = c( "+", "-" ),
                                            true = c( "+", "-" ) ) )
}

#### Derived matrics
precision <- function( confusion_cube )
## Computes the precision, which is TP / ( TP + FP )
  apply( confusion_cube, 3, function( T ) T["+","+"]/( T["+","+"]+T["+","-"] ) )

recall <- function( confusion_cube )
## Computes the recall, which is TP / ( TP + TN )
  apply( confusion_cube, 3, function( T ) T["+","+"]/( T["+","+"]+T["-","+"] ) )

F_measure <- function( confusion_cube ) {
## Compute the F-score : the harmonic average of precision and recall
  prc <- precision( confusion_cube )
  rcl <- recall( confusion_cube )
  2 * ( 1/prc + 1/rcl )
}

specificity <- function( confusion_cube )
## Specificity is the ratio of TN to ( TN + FP )
  apply( confusion_cube, 3, function( T ) T["-","-"]/( T["-","-"]+T["+","-"] ) )

fdr <- function( confusion_cube )
## Computes the Flase Discovery Rate, which is just one
##  minus precision.
  1 - precision( confusion_cube )


if( FALSE ) {
  matrix_ <- confusion( trainLabels, predictedLabels )
  confusion_cube <-
        simplify2array(
              sapply( dimnames( matrix_ )$true, confusion2x2,
                      matrix_, simplify = FALSE ) )

  precision( confusion_cube )
  recall( confusion_cube )
  F_measure( confusion_cube )
  specificity( confusion_cube )
  fdr( confusion_cube )

}
