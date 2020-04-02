export TRAINING_DATA=input/train_folds.csv
export TEST_DATA=input/test.csv
export MODEL=$1

#create folds run before everything, run only once
#C:\\ProgramData\\Anaconda3\\envs\\TF2.0\\python -m src.create_folds

#train the model on folds
#FOLD=0 C:\\ProgramData\\Anaconda3\\envs\\TF2.0\\python -m src.train
#FOLD=1 C:\\ProgramData\\Anaconda3\\envs\\TF2.0\\python -m src.train
#FOLD=2 C:\\ProgramData\\Anaconda3\\envs\\TF2.0\\python -m src.train
#FOLD=3 C:\\ProgramData\\Anaconda3\\envs\\TF2.0\\python -m src.train
#FOLD=4 C:\\ProgramData\\Anaconda3\\envs\\TF2.0\\python -m src.train

#comment below while training
#C:\\ProgramData\\Anaconda3\\envs\\TF2.0\\python -m src.predict
