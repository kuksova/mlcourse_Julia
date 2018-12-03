using DataFrames
using DecisionTree
# step 1. Data read
df = readtable("telecom_churn.csv");
states = df[:State];
churn = permutedims(df[:Churn]);
df_not_states = delete!(df, :State)
df_not_states = delete!(df_not_states, :Churn)
#Convert from character to integer
#####
df_not_states[:International_plan] = map(x -> x[1], df_not_states[:International_plan]);
df_not_states[:Voice_mail_plan] = map(x -> x[1], df_not_states[:Voice_mail_plan]);
df_not_states[:International_plan] = map(Int, df_not_states[:International_plan]);
df_not_states[:Voice_mail_plan] = map(Int, df_not_states[:Voice_mail_plan]);


# step 2. Split data 70%; 30%
# need to make random split
#X_train = first(df_not_states, size(df_not_states,1)*0.7);

n = convert(Int, round(size(df_not_states,1)*0.7));
X_train = df_not_states[1:n, :];
X_test = df_not_states[n+1:end, :];

yTrain = map(x -> x[1], churn[1:n, :]);
#Convert from character to integer
yTrain = map(Int, yTrain);
yTest = map(x -> x[1], churn[n+1:end, :]);
#Convert from character to integer
yTest = map(Int, yTest);

X_train = convert(Array, df_not_states[1:n, :]);
X_test = convert(Array, df_not_states[n+1:end, :]);

# step 3.
model = build_forest(yTrain, X_train, 17, 50, 1.0);

# #Get predictions for test data
predTest = apply_forest(model, X_train)

loofCvAccuracy = mean(predTest  .== yTrain)
