# dir.create(path = Sys.getenv("R_LIBS_USER"), showWarnings = FALSE, recursive = TRUE)
# install.packages("seriation", lib = Sys.getenv("R_LIBS_USER"))
# install.packages("seriation")
suppressPackageStartupMessages(library(seriation))
suppressPackageStartupMessages(library(biclust))
# set.seed(123)

r_seriation_dist <- function(mat, method="OLO"){
    d <- as.dist(mat)
    order <- seriate(d, method)
    pimage(d)
    pimage(d,  order)
    return (get_order(order))
}

r_seriation_mat <- function(mat, method="PCA"){
    order <- seriate(mat, method)
    return (get_order(order))
}

r_seriation_GA <- function(mat, method="GA"){
    d <- as.dist(mat)
    register_GA()
    gaperm_mixedMutation(ismProb = 0.8)
    order <- seriate(d, method)
    pimage(d)
    pimage(d,  order)
    return (get_order(order))
}

r_seriation_DendSer <- function(mat, method="DendSer"){
    d <- as.dist(mat)
    register_DendSer()
    order <- seriate(d, method)
    pimage(d)
    pimage(d,  order)
    return (get_order(order))
}

r_criterion  <- function(mat, method=NULL){
    d <- as.dist(mat)
    return (criterion(d, method=method))
}

r_test_criterion <- function(mat, method=NULL, order=NULL){
    d <- as.dist(mat)
    if(is.null(order)){
        o <- seriate(mat, order)
        return (criterion(d, order=o, method=method))
    }
    else{
        return (criterion(d, method=method))
    }
}

r_biclust <- function(mat, method="BCPlaid"){
    print((biclusternumber(biclust(mat, method))))
    return (biclusternumber(biclust(mat, method)))
}

