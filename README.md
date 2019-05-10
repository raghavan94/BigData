Tensorflow must be installed first to run the application. Use the following command below to install tensorflow.

#sudo pip install tensorflow

To monitor the CPU/Memory/Network usage of each Virtual Machine during model training, we used a tool called dstat. It can be installed by using the following command:

#sudo apt-get install dstat



Cluster Specifications:

The model can be trained in 3 different cluster configurations - single, cluster, cluster2.


1. Cluster Mode - Single

To start the server in this mode use the below command

./run.sh /code/basic_rnn_static.py single
./run.sh /code/lstm_rnn_dynamic.py single





2. Cluster Mode - cluster2 (1PS, 3 Worker Nodes)

In a distributer environment, we can train the model both synchronous and asynchronous.


./run.sh /code/distributed_basic_rnn_static_clusterspec cluster2 
./run.sh /code/distributed_lstm_rnn_dynamic_clusterspec cluster2 
./run.sh /code/distributed_basic_rnn_static_sync cluster2 
./run.sh /code/distributed_lstm_rnn_dynamic_sync cluster2 

3. We ran the script for different hyperparameters and monitor their differences.