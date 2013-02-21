# local level model
testHMKF1fromR <- function(){
  set.seed(8888)
  numSteps <- 100
  sx <- 1
  sy <- 1
  x <- rnorm(n=numSteps,mean=0,sd=sx)
  x <- cumsum(x)
  y <- x + rnorm(n=numSteps,mean=0,sd=sy)
  y <- matrix(data=y,nrow=1)
  
  a0 <- 0
  P0 <- matrix(1)
  TT  <- matrix(1)
  ZZ  <- matrix(1)
  HH  <- matrix(1)
  GG  <- matrix(0.5)
  result <- HandMadeKalmanFilterConstantCoeffCpp(a0=a0,P0=P0,TT,Z=ZZ,HH=HH,GG=GG,yt=y)

  plot(y[1,])
  points(result$x, col="blue")
  points(result$x + sqrt(result$V), col="green")
  points(result$x - sqrt(result$V), col="green")
}
