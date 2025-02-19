library(GPArotation)
library(parallel)
set.seed(2024)
rot_fun <- function(params) {
  L_efa = params$L
  iter = params$iter
  result = bifactorQ(L_efa, Tmat=diag(ncol(L_efa)), normalize=FALSE, eps=1e-5, maxit=5000,randomStarts=50)
  
  L = result$loadings
  Psi = result$Phi
  return(list(L = L, Psi = Psi, iter = iter))
}

inputs <- list()

case <- 'case1'
J <- 30
G <- 5
n <- 2000

Epoch<- 100
for (i in 1:Epoch){
  load_name = paste0('BF_minor_sim/Rot/L_efa_', J, '_', G, '_', n, '_', case, '_', i-1, '.csv')
  data = read.csv(load_name, header = TRUE, sep = ",")
  data = as.matrix(data)
  data = data[,-1]
  inputs[[i]] = list(L=data,iter=i)
}

print(inputs[[1]])
cl <- makeCluster(min(detectCores() - 1, 20))  

clusterEvalQ(cl, library(GPArotation))

clusterExport(cl, varlist = c("rot_fun"))

group_size <- 10
num_groups <- ceiling(Epoch / group_size)
results <- list()
for (group_idx in 1:num_groups) {
  start_idx <- (group_idx - 1) * group_size + 1
  end_idx <- min(group_idx * group_size, Epoch)
  group_inputs <- inputs[start_idx:end_idx]
  
  group_results <- parLapply(cl, group_inputs, rot_fun)
  results <- c(results, group_results)  
}

stopCluster(cl) 
for (i in seq_along(results)) {
  iter <- results[[i]]$iter
  Psi_mat<-results[[i]]$Psi
  L_mat <- results[[i]]$L
  L_name = paste0('BF_minor_sim/Rot/L_bf_', J, '_', G, '_', n, '_', case, '_', iter-1, '.csv')
  Psi_name = paste0('BF_minor_sim/Rot/Psi_bf_', J, '_', G, '_', n, '_', case, '_', iter-1, '.csv')
  
  write.csv(L_mat, file = L_name, row.names = FALSE)
  write.csv(Psi_mat, file = Psi_name, row.names = FALSE)
}