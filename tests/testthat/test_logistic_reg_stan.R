library(testthat)
library(parsnip)
library(rlang)

context("execution tests for stan logistic regression")


data("lending_club")
lending_club <- head(lending_club, 200)
lc_form <- as.formula(Class ~ log(funded_amnt) + int_rate)
num_pred <- c("funded_amnt", "annual_inc", "num_il_tl")
lc_basic <- logistic_reg(others = list(seed = 1333, chains = 1))
ctrl <- fit_control(verbosity = 1, catch = FALSE)
caught_ctrl <- fit_control(verbosity = 1, catch = TRUE)
quiet_ctrl <- fit_control(verbosity = 0, catch = TRUE)

xy_fit <- fit(
  logistic_reg(others = list(seed =  11, chains = 1)),
  engine = "stan",
  control = ctrl,
  x = lending_club[, num_pred],
  y = lending_club$Class
)

print(xy_fit)

test_that('stan_glm execution', {

  expect_error(
    res <- fit(
      lc_basic,
      funded_amnt ~ term,
      data = lending_club,
      engine = "stan",
      control = ctrl
    )
  )

  stan_xy_catch <- fit(
    lc_basic,
    engine = "stan",
    control = caught_ctrl,
    x = lending_club[, num_pred],
    y = lending_club$total_bal_il
  )
  expect_true(inherits(stan_xy_catch$fit, "try-error"))

})


test_that('stan_glm prediction', {

  xy_fit <- fit(
    logistic_reg(others = list(seed =  11, chains = 1)),
    engine = "stan",
    control = ctrl,
    x = lending_club[, num_pred],
    y = lending_club$Class
  )

  xy_pred <-
    predict(xy_fit$fit,
            newdata = lending_club[1:7, num_pred])
  xy_pred <- ifelse(xy_pred >= 0.5, "good", "bad")
  xy_pred <- factor(xy_pred, levels = levels(lending_club$Class))
  xy_pred <- unname(xy_pred)

  expect_equal(xy_pred, predict_class(xy_fit, lending_club[1:7, num_pred]))

  res_form <- fit(
    logistic_reg(others = list(seed =  11, chains = 1)),
    Class ~ log(funded_amnt) + int_rate,
    data = lending_club,
    engine = "stan",
    control = ctrl
  )

  form_pred <-
    predict(res_form$fit,
            newdata = lending_club[1:7, c("funded_amnt", "int_rate")])
  form_pred <- unname(form_pred)
  form_pred <- ifelse(form_pred >= 0.5, "good", "bad")
  form_pred <- factor(form_pred, levels = levels(lending_club$Class))
  expect_equal(form_pred, predict_class(res_form, lending_club[1:7, c("funded_amnt", "int_rate")]))
})