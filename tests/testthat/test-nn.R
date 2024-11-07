test_that("neuron works as expected", {
  .[params, forward] <- neuron(2, identity)

  expect_equal(length(unlist(params)), 3)

  x <- list(Value(1), Value(2))
  y <- forward(x, params)

  expected <- params$w[[1]]@data * 1 + params$w[[2]]@data * 2 + params$b@data  

  expect_equal(y@data, expected)
})

test_that("linear layer works as expected", {
  .[params, forward] <- layer(2, 2, act = identity)

  expect_equal(length(unlist(params)), 6)

  x <- list(Value(1), Value(2))
  y <- forward(x, params)

  expected <- params[[1]]$w[[1]]@data * 1 + params[[1]]$w[[2]]@data * 2 + params[[1]]$b@data  
  expect_equal(y[[1]]@data, expected)

  expected <- params[[2]]$w[[1]]@data * 1 + params[[2]]$w[[2]]@data * 2 + params[[2]]$b@data  
  expect_equal(y[[2]]@data, expected)
})

test_that("mlp works as expected", {

  .[params, forward] <- mlp(2, c(2, 1))

  expect_equal(length(unlist(params)), 9)

  x <- list(Value(1), Value(2))
  y <- forward(x, params)

  expect_length(y, 1)
})


test_that("linear regression", {
  skip_if_not_installed("torch")

  .[init_params, model] <- layer(1, 1, act = identity)

  expect_equal(length(unlist(init_params)), 2)

  x <- runif(100)
  y <- 2 * x + 1

  loss <- function(params) {
    loss <- Value(0)
    for (i in seq_along(x)) {
      y_hat <- model(list(Value(x[i])), params)[[1]]
      loss <- loss + ((y_hat - Value(y[i])) ^ Value(2))
    }
    loss / Value(100)
  }

  apply_grads <- function(params, grads, lr = 0.01) {
    purrr::map2(params, grads, function(p, g) {
      if (!is.list(p) && !is.list(g)) {
        Value(p@data - lr * g)
      } else {
        apply_grads(p, g, lr)
      }
    })
  }

  params <- init_params
  for (i in 1:10) {
    .[value, .[grads]] <- value_and_grad(loss)(params)
    params <- apply_grads(params, grads, lr = 0.001)
  }


  # result with torch
  model_tch <- torch::nn_linear(1,1)
  model_tch$to(dtype = torch::torch_double())
  torch::with_no_grad({
    model_tch$weight$set_(torch::torch_tensor(init_params[[1]]$w[[1]]@data, dtype="float64")$view_as(model_tch$weight))
    model_tch$bias$set_(torch::torch_tensor(init_params[[1]]$b@data, dtype="float64")$view_as(model_tch$bias))
  })
  
  for (i in 1:10) {
    y_hat <- model_tch(torch::torch_tensor(x, dtype="float64")$view(c(-1,1)))
    loss <- (((y_hat - torch::torch_tensor(y, dtype="float64")$view(c(-1,1)))^2))$sum()/100
    loss$backward()

    torch::with_no_grad({
      model_tch$weight$sub_(0.001 * model_tch$weight$grad)
      model_tch$bias$sub_(0.001 * model_tch$bias$grad)
      model_tch$zero_grad()
    })
  }

  expect_equal(loss$item(), as.numeric(value), tol = 1e-5)

})



# options(warn=2)
