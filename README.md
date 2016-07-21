# ATIS-Slot-Filling

Task Information:
- Data is the Airline Travel Information System (ATIS) dataset collected by DARPA. Contains text information where words are to be understood labelled. Eg.
   
   Input (words) :	 show |	flights |	from |	Boston |	to |	New	  |  York  |	today
   
   Output (labels):   O	  |    O	  |   O	 |  B-dept |	 O |	B-arr |	 I-arr |	B-date

- Completed said task by achieving Spoken Language Understanding through the implementation of a Recurrent Neural Network
- Made use of a sliding context window, word embeddings, an Elman Recurrent Network and standard perl script Conlevall.pl for evaluation

Information for Execution:
- Run RecurrentNeuralNetwork.py from Terminal/IDE
- Conlevall.pl is used to check the accuracy, has been included in the repository
- .pkl version of the dataset is also included
- Both the above should be in the same directory as the python script
