test_that("can multiply numbers and get derivatives", {
  x <- Value(2)
  y <- Value(3)

  f <- function(x, y) {
    x * y
  }

  .[value, .[grad_x, grad_y]] = value_and_grad(f)(x = x, y = y)
  expect_equal(value, 6)
  expect_equal(grad_x, 3)
  expect_equal(grad_y, 2)
})

test_that("add and multiply", {
  x <- Value(2)
  y <- Value(3)
  z <- Value(4)

  f <- function(x, y, z) {
    x * y + z
  }

  .[value, .[grad_x, grad_y, grad_z]] = value_and_grad(f)(x = x, y = y, z = z)
  expect_equal(value, 10)
  expect_equal(grad_x, 3)
  expect_equal(grad_y, 2)
  expect_equal(grad_z, 1)
})

test_that("slightly more complex add and multiply", {
  x <- Value(2)
  y <- Value(3)
  z <- Value(4)

  f <- function(x, y, z) {
    x * y + x * z
  }

  .[value, .[grad_x, grad_y, grad_z]] = value_and_grad(f)(x = x, y = y, z = z)
  expect_equal(value, 14)
  expect_equal(grad_x, 7)
  expect_equal(grad_y, 2)
  expect_equal(grad_z, 2)
})

test_that("division works", {
  x <- Value(2)
  y <- Value(3)

  f <- function(x, y) {
    x / y
  }

  .[value, .[grad_x, grad_y]] = value_and_grad(f)(x = x, y = y)
  expect_equal(value, 2/3)
  expect_equal(grad_x, 1/3)
  expect_equal(grad_y, -2/9)
})

test_that("a + a", {
  x <- Value(2)

  f <- function(x) {
    x + x
  }

  .[value, .[grad_x]] = value_and_grad(f)(x = x)
  expect_equal(value, 4)
  expect_equal(grad_x, 2)
})

test_that("slightly more complex example", {

  f <- function(x) {
    z <- x * 2 + 2 + x
    q <- relu(z) + z * x
    h <- relu(z * z)
    y <- h + q + q * x
    y
  }

  .[value, .[grad_x]] <- value_and_grad(f)(x = Value(-4))

  skip_if_not(rlang::is_installed("torch"))

  x <- torch::torch_tensor(-4, requires_grad = TRUE)
  z <- 2 * x + 2 + x
  q <- z$relu() + z * x
  h <- (z * z)$relu()
  y <- h + q + q * x
  y$backward()

  value_tch <- as.numeric(y)
  grad_x_tch <- as.numeric(x$grad)
  
  expect_equal(grad_x, grad_x_tch)
  expect_equal(value, value_tch)
})

