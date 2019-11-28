Statisitical Programming - Project 2
================
vanessafun

Due by 11:59 pm on Friday, December 6th.

## Question 1

define a density function of bivariate gaussian,a random-walk MCMC
algorithm is used to simulates observations from this density

``` r
library(ggplot2)
library(magrittr)
dbigaussian <- function(x1, x2) {
  sigma <- matrix(c(1, 0.5, 0.5, 1), 2, 2)
  u <- c(1, 1)
  return(as.numeric(exp(
    -0.5 * t(c(x1, x2) - u) %*% solve(sigma) %*% (c(x1, x2) - u) / (2 * pi) /sqrt(0.75)
  )))
}
n <- 10000 #number of samples to draw
x1s <- numeric(n)
x2s <- numeric(n)
x1s[1] <- 0 #initial value
x2s[1] <- 0 #initial value
for (i in 2:n) {
  x1 <- x1s[i - 1] + rnorm(1, 0, 1)
  x2 <- x2s[i - 1] + rnorm(1, 0, 1)
  a <- dbigaussian(x1, x2) / dbigaussian(x1s[i - 1], x2s[i - 1])
  if (runif(1) < a) {
    x1s[i] <- x1
    x2s[i] <- x2
  } else {
    x1s[i] <- x1s[i - 1]
    x2s[i] <- x2s[i - 1]
  }
}
x1s <- x1s[-(1:100)]
x2s <- x2s[-(1:100)]
df = data.frame(x1s, x2s)
ggplot(df, aes(x = x1s, y = x2s)) +
  geom_density_2d() +
  labs(title = "contour of bigaussian")
```

![](proj2_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

## Question 2

### 2.1

variance of asmple is 1.132592,variance of Student-t(20) is 1.11, they
are pretty close,which means those samples are genuine draws from the
correct Student-t(20) distribution.

``` r
set.seed(2)
dstudentt <- function(y) {
  v <- 20
  return(gamma((v + 1) / 2) / (sqrt(v * pi) * gamma(v / 2)) *
           (1 + (y ^ 2) / v) ^ (-(v + 1) / 2))
}
n <- 10000 #number of samples to draw
ys <- numeric(n)
ys[1] <- 0 #initial value
for (i in 2:n) {
  y <- ys[i - 1] + rnorm(1, 0, 1)
  a <- dstudentt(y) / dstudentt(ys[i - 1])
  if (runif(1) < a) {
    ys[i] <- y
  } else {
    ys[i] <- ys[i - 1]
  }
}
ys <- ys[-(1:100)]
print(var(ys))
```

    ## [1] 1.132592

``` r
print(20 / 18)
```

    ## [1] 1.111111

### 2.2

the sample variance(which is 2.758434) is not close nor equal to
student-t(3) theoretical variance(which is 3), one possible explanation
is that convergence dosen’t happen before the 10000th sample or happened
really late, for example, untill the 7000th sample this MCMC converged,
another explanation is that this MCMC’s convergence is not Student-t(3)

``` r
set.seed(2)
dstudentt <- function(y) {
  v <- 3
  return(gamma((v + 1) / 2) / (sqrt(v * pi) * gamma(v / 2)) *
           (1 + (y ^ 2) / v) ^ (-(v + 1) / 2))
}
n <- 10000 #number of samples to draw
ys <- numeric(n)
ys[1] <- 0 #initial value
for (i in 2:n) {
  y <- ys[i - 1] + rnorm(1, 0, 1)
  a <- min(1, dstudentt(y) / dstudentt(ys[i - 1]))
  if (runif(1) < a) {
    ys[i] <- y
  } else {
    ys[i] <- ys[i - 1]
  }
}
ys <- ys[-(1:100)]
print(var(ys))
```

    ## [1] 2.758434

``` r
print(3 / 1)
```

    ## [1] 3

## Question 3

``` r
y = scan("data/eventtimes.csv", sep=",")
likli<-function(k){
  n<-100
  lamda<-2
  multi<-1
  for (i in 1:n){
    multi<-multi*y[i]
  }
  sumup<-0
  for (j in 1:n){
    sumup<-sumup + (y[i])^k
  }
  return(
    exp(-sumup/(lamda^k))*(multi^(k-1))*(k^n)/(lamda^(n*k))
  )
}
```

### 3.1

### 3.2

### 3.3

### 3.4

### 3.5

### 3.6

### 3.7

### 3.8
