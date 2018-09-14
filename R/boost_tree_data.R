
boost_tree_arg_key <- data.frame(
  xgboost = c("max_depth", "nrounds", "eta", "colsample_bytree", "min_child_weight", "gamma", "subsample"),
  C5.0 =    c(         NA,  "trials",    NA,                NA,         "minCases",       NA,    "sample"),
  catboost =c("depth", "iterations", "learning_rate", "rsm", NA, NA, NA),
  stringsAsFactors = FALSE,
  row.names =  c("tree_depth", "trees", "learn_rate", "mtry", "min_n", "loss_reduction", "sample_size")
)

boost_tree_modes <- c("classification", "regression", "unknown")

boost_tree_engines <- data.frame(
  xgboost =    rep(TRUE, 3),
  C5.0    =    c(            TRUE,        FALSE,      TRUE),
  catboost =   rep(TRUE, 3),
  row.names =  c("classification", "regression", "unknown")
)

###################################################################

xgb_train <- function(
  x, y,
  max_depth = 6, nrounds = 15, eta  = 0.3, colsample_bytree = 1,
  min_child_weight = 1, gamma = 0, subsample = 1, ...) {

  num_class <- if (length(levels(y)) > 2) length(levels(y)) else NULL

  if (is.numeric(y)) {
    loss <- "reg:linear"
  } else {
    lvl <- levels(y)
    y <- as.numeric(y) - 1
    if (length(lvl) == 2) {
      loss <- "binary:logistic"
    } else {
      loss <- "multi:softprob"
    }
  }

  if (is.data.frame(x))
    x <- as.matrix(x) # maybe use model.matrix here?

  n <- nrow(x)
  p <- ncol(x)

  if (!inherits(x, "xgb.DMatrix"))
    x <- xgboost::xgb.DMatrix(x, label = y, missing = NA)
  else
    xgboost::setinfo(x, "label", y)

  # translate `subsample` and `colsample_bytree` to be on (0, 1] if not
  if(subsample > 1)
    subsample <- subsample/n
  if(subsample > 1)
    subsample <- 1

  if(colsample_bytree > 1)
    colsample_bytree <- colsample_bytree/p
  if(colsample_bytree > 1)
    colsample_bytree <- 1

  arg_list <- list(
    eta = eta,
    max_depth = max_depth,
    gamma = gamma,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    subsample = subsample
  )

  # eval if contains expressions?

  main_args <- list(
    data = quote(x),
    params = arg_list,
    nrounds = nrounds,
    objective = loss
  )
  if (!is.null(num_class))
    main_args$num_class <- num_class

  call <- make_call(fun = "xgb.train", ns = "xgboost", main_args)

  # override or add some other args
  others <- list(...)
  others <-
    others[!(names(others) %in% c("data", "weights", "nrounds", "num_class", names(arg_list)))]
  if (length(others) > 0)
    for (i in names(others))
      call[[i]] <- others[[i]]

  eval_tidy(call, env = current_env())
}

#' @importFrom stats binomial
xgb_pred <- function(object, newdata, ...) {
  if (!inherits(newdata, "xgb.DMatrix")) {
    newdata <- as.matrix(newdata)
    newdata <- xgboost::xgb.DMatrix(data = newdata, missing = NA)
  }

  res <- predict(object, newdata, ...)

  x = switch(
    object$params$objective,
    "reg:linear" =, "reg:logistic" =, "binary:logistic" = res,
    "binary:logitraw" = stats::binomial()$linkinv(res),
    "multi:softprob" = matrix(res, ncol = object$params$num_class, byrow = TRUE),
    res
  )
  x
}

C5.0_train <- function(x, y, weights = NULL, trials = 15, minCases = 2, sample = 0, ...) {
  other_args <- list(...)
  protect_ctrl <- c("minCases", "sample")
  protect_fit <- "trials"
  other_args <- other_args[!(other_args %in% c(protect_ctrl, protect_fit))]
  ctrl_args <- other_args[names(other_args) %in% names(formals(C50::C5.0Control))]
  fit_args <- other_args[names(other_args) %in% names(formals(C50::C5.0.default))]

  ctrl <- expr(C50::C5.0Control())
  ctrl$minCases <- minCases
  ctrl$sample <- sample
  for(i in names(ctrl_args))
    ctrl[[i]] <- ctrl_args[[i]]

  fit_call <- expr(C50::C5.0(x = x, y = y))
  fit_call$trials <- trials
  fit_call$control <- ctrl
  if(!is.null(weights))
    fit_call$weights <- quote(weights)

  for(i in names(fit_args))
    fit_call[[i]] <- fit_args[[i]]
  eval_tidy(fit_call)
}


# Catboost definition -----------------------------------------------------

catboost_train <- function(
  x, y,
  depth = 6, iterations = 500, learning_rate  = 0.03, rsm = 1,
  logging_level = 'Silent', thread_count = 1, od_pval = 1E-3, ...) {

  if (is.numeric(y)) {
    loss <- "RMSE"
    eval_loss <- "RMSE"
  } else {

    if(is.logical(y)){
      lvl <- c(F,T)
      dif <- 0
    } else {
      lvl <- levels(y)
      dif <- 1
    }

    y <- as.integer(y) - dif

    if (length(lvl) == 2) {
      loss <- "Logloss"
      eval_loss <- "AUC"
    } else {
      loss <- "CrossEntropy"
      eval_loss <- "MultiClass"
    }
  }

  n <- nrow(x)
  p <- ncol(x)

  x <- catboost::catboost.load_pool(x, label = y, feature_names = as.list(colnames(x)) )

  # `colsample_bytree` to be on (0, 1] if not

  if(rsm > 1)
    rsm <- rsm/p
  rsm <- min(1, rsm)

  arg_list <- list(
    depth = depth,
    iterations = iterations,
    learning_rate = learning_rate,
    rsm = rsm,
    loss_function = loss,
    eval_metric = eval_loss,
    allow_writing_files = FALSE,
    logging_level = logging_level,
    thread_count = thread_count,
    od_pval = od_pval)

  # eval if contains expressions?
  others <- list(...)

  # Not really proud of this.
  if (all(c("test_data","test_label") %in% others)) {
    if(loss != "RMSE"){
      test_label <- as.integer(others$test_label) - dif
    }

    tpool <- catboost::catboost.load_pool(as.data.frame(others$test_data)[,colnames(x)],
                                          label = test_label,
                                          feature_names = as.list(colnames(x)) )
  } else {
    tpool <- NULL
  }


  others <-
    others[!(names(others) %in% c("learn_pool", "test_pool", "test_data", "test_label",
                                  "subsample", names(arg_list)))]

  main_args <- list(
    learn_pool = quote(x),
    test_pool = quote(tpool),
    params = arg_list)

  call <- make_call(fun = "catboost.train", ns = "catboost", main_args)

  # override or add some other args
  if (length(others) > 0)
    for (i in names(others))
      call[[i]] <- others[[i]]

  ret <- eval_tidy(call, env = rlang::current_env())

  if (loss != "RMSE") {
    ret$levels <- lvl
  }

  ret
}

catboost_pred <- function(object, newdata, pred_type, ...) {

  newdata <- catboost::catboost.load_pool(newdata, feature_names = as.list(colnames(newdata)))

  res <- catboost::catboost.predict(object, newdata, verbose = FALSE,
                                    prediction_type = pred_type, ...)

  if(pred_type == "Probability"){
    if (!is.data.frame(res) & !is.matrix(res)){
      res <- cbind( res, 1-res)
    }

    res <- as.data.frame(res)
    colnames(res) <- object$levels
  } else if (pred_type == "Class") {
    res <- object$levels[res+1]
  }

  res

}


###################################################################


boost_tree_xgboost_data <-
  list(
    libs = "xgboost",
    fit = list(
      interface = "matrix",
      protect = "data",
      func = c(pkg = NULL, fun = "xgb_train"),
      defaults =
        list(
          nthread = 1,
          verbose = 0
        )
    ),
    pred = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "xgb_pred"),
      args =
        list(
          object = quote(object$fit),
          newdata = quote(new_data)
        )
    ),
    classes = list(
      pre = NULL,
      post = function(x, object) {
        if (is.vector(x)) {
          x <- ifelse(x >= 0.5, object$lvl[2], object$lvl[1])
        } else {
          x <- object$lvl[apply(x, 1, which.max)]
        }
        x
      },
      func = c(pkg = NULL, fun = "xgb_pred"),
      args =
        list(
          object = quote(object$fit),
          newdata = quote(new_data)
        )
    ),
    prob = list(
      pre = NULL,
      post = function(x, object) {
        if (is.vector(x)) {
          x <- tibble(v1 = 1 - x, v2 = x)
        } else {
          x <- as_tibble(x)
        }
        colnames(x) <- object$lvl
        x
      },
      func = c(pkg = NULL, fun = "xgb_pred"),
      args =
        list(
          object = quote(object$fit),
          newdata = quote(new_data)
        )
    ),
    raw = list(
      pre = NULL,
      func = c(fun = "xgb_pred"),
      args =
        list(
          object = quote(object$fit),
          newdata = quote(new_data)
        )
    )
  )


boost_tree_C5.0_data <-
  list(
    libs = "C50",
    fit = list(
      interface = "data.frame",
      protect = c("x", "y", "weights"),
      func = c(pkg = NULL, fun = "C5.0_train"),
      defaults = list()
    ),
    classes = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = quote(object$fit),
        newdata = quote(new_data)
      )
    ),
    prob = list(
      pre = NULL,
      post = function(x, object) {
        as_tibble(x)
      },
      func = c(fun = "predict"),
      args =
        list(
          object = quote(object$fit),
          newdata = quote(new_data),
          type = "prob"
        )
    ),
    raw = list(
      pre = NULL,
      func = c(fun = "predict"),
      args = list(
        object = quote(object$fit),
        newdata = quote(new_data)
      )
    )
  )

boost_tree_catboost_data <-
  list(
    libs = "catboost",
    fit = list(
      interface = "data.frame",
      protect = c("x", "y"),
      func = c(pkg = NULL, fun = "catboost_train"),
      defaults = list()
    ),
    pred = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "catboost_pred"),
      args =
        list(
          object = quote(object$fit),
          newdata = quote(new_data),
          pred_type = "RawFormulaVal"
        )
    ),
    classes = list(
      pre = NULL,
      post = NULL,
      func = c(pkg = NULL, fun = "catboost_pred"),
      args =
        list(
          object = quote(object$fit),
          newdata = quote(new_data),
          pred_type = "Class"
        )
    ),
    prob = list(
      pre = NULL,
      post = NULL,
      func = c(pkg = NULL, fun = "catboost_pred"),
      args =
        list(
          object = quote(object$fit),
          newdata = quote(new_data),
          pred_type = "Probability"
        )
    ),
    raw = list(
      pre = NULL,
      func = c(fun = "catboost_pred"),
      args =
        list(
          object = quote(object$fit),
          newdata = quote(new_data),
          pred_type = "RawFormulaVal"
        )
    )
  )
