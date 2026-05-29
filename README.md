
This README file provides instructions for installing and running the code associated with the paper:

**[Decoding charge sequence control of polyelectrolyte interactions and electrocapacitive
behavior through high-throughput simulations and interpretable machine learning]**
*Yan Sui,† Xueying Yuan,† and Xian Kong*

---

## 1. Contents
── project/ # data; script
── results/ # out files
── README.md # this text


## 2. System Requirements

### Hardware
- CPU: Any modern multi-core processor (e.g., Intel Xeon or AMD EPYC)
- RAM: 16 GB or more recommended
- Storage: 10 GB free space

### Software
- Operating System: Linux (Ubuntu 20.04/22.04) 
- Dependencies: See `requirements.txt`

## 3. Installation

3.1 Set up environment (using conda)

conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt


## 4.  Running the Code

4.1 Data 

1.csv  #data 

4.2 Model training

Train the CNN model:

python project/cnn.py  
  
4.3  using the shell script 

bash project/sy.sh

## 5. Expected Output

After successful execution, you should see:

./results/final_r2_mse_kn_{k_n}_ks_{k_s}.txt   --Final training and validation MSE and R² scores. 
./results/train-pre_kn_{k_n}_ks_{k_s}.txt	--Predicted values for the training set (one value per line).
./results/test-pre_kn_{k_n}_ks_{k_s}.txt	--Predicted values for the validation/test set (one value per line).
./results/y-train_kn_{k_n}_ks_{k_s}.txt	--Ground truth labels for the training set.
./results/y-test_kn_{k_n}_ks_{k_s}.txt	--Ground truth labels for the validation/test set.
./results/ori_weight.txt	--Initial convolutional layer weights before training.
./results/final_weights_kn_{k_n}_ks_{k_s}.txt	--Final convolutional layer weights after training (last iteration).
./results/final_activation_kn_{k_n}_ks_{k_s}.txt	--Final convolutional layer activations (output of Conv1d) for the training set. Each row corresponds to one sample, with flattened feature maps.
./results/x_train_kn_{k_n}_ks_{k_s}.txt	--Training input data (space-separated format).
./results/x_train_ver_kn_{k_n}_ks_{k_s}.txt	--Training input data (one value per line, column format).
./results/combined_train_test_pred.png	--Scatter plot of true vs. predicted values for both training and test sets, including the identity line (y=x) and performance metrics (R² and MSE).
./results/model.pth	--Saved PyTorch model state dictionary (weights and architecture).

## 6. Troubleshooting
Issue	  Solution
ImportError: No module named xxx	 Run pip install -r requirements.txt
Out of memory	Reduce batch size in config file
