library(hydra)
library(ape)
library(MASS)

if(!require('hydra')) {
  install.packages('hydra')
  library('hydra')
}

if(!require('ape')) {
  install.packages('ape')
  library('ape')
}

if(!require('MASS')) {
  install.packages('MASS')
  library('MASS')
}

options(echo=TRUE) # if you want see commands in output file
args <- commandArgs(trailingOnly = TRUE)


tree<-read.tree(args[1]) 
Nall <- dim(D)[1]
X1 <- D
X2 <- acosh(exp(D))

X2.hydra <- hydraPlus(X2, dim=M, curvature=1, alpha=1, equi.adj=0, control=list(return.dist=1, isotropic.adj=FALSE))
Z2 <- X2.hydra$r * X2.hydra$directional
X2.hydra$dist <- hydra:::get.distance(X2.hydra$r, X2.hydra$directional, X2.hydra$curvature)
write_csv(relaxed_output,paste(str_trim(args[2]),"/r_lasso_relaxed.csv", sep=""))

tree$mrca <- mrca(tree)
D <- dist.nodes(tree)

