library(np)
# function npindex: first coefficient is 1

setwd('~/Desktop/to_do/pricing_revision')
Loan_df <- read.csv(file = 'Loan_preprocessed.csv')
Loan_df <- Loan_df[,-1] # drop index column
colnames(Loan_df) <- c('y', 'FICO', 'loan', 'onemonth', 'competition', 'price')
for (i in c(2:6)){  # standardize column
    col_mean = mean(Loan_df[, i])
    col_sd = sd(Loan_df[, i])
    Loan_df[, i] = (Loan_df[, i] - col_mean) / col_sd
}

row_ind = sample(nrow(Loan_df), 5000)
Loan_df1 = Loan_df[row_ind, ]

# compute theta
bw <- npindexbw(formula= y ~ price + FICO + loan + onemonth + competition,
                data = Loan_df1, method = 'ichimura')
model <- npindex(bws=bw, gradients=TRUE)#model
#g = model$grad
#g1 = (g - mean(g))/sd(g)
#hist(g1, breaks = 10, freq = F)
