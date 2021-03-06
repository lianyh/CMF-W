# CMF-W
--------------------------------
*Prerequisite*
--------------------------------
Please install:<br/>
python >= 3.6 version<br/>
Tensorflow >= 1.12.0<br/>
scikit-learn >= 0.20.2<br/>

Command:
------------------------------
python cmfw.py data/F1_F2_SL_binary data/F1_F2_coexpr_for_train data/F1_F2_me_for_train data/F1_F2_pathway_for_train data/F1_F2_ppi_for_train data/F1_F2_proteincomplex_for_train data/F1_F2_index_for_train data/F3_index_for_test 332<br/>

1st column: training set with {1,0} binary value, 1=SL, 0=non-SL (gene(row) * gene(col) matrix)<br/>
2nd column: co-expression training set with co-expression score (gene(row) * gene(col) matrix)<br/>
3rd column: mutual exclusivity(ME) training set with ME score (gene(row) * gene(col) matrix)<br/>
4th column: pathway training set with {1,0} binary value, 1=co-membership, 0=non co-membership (gene(row) * gene(col) matrix)<br/>
5th column: ppi training set with {ppi interaction score} <br/>
6th column: protein-complex training set with {1,0} binary value, 1=complex co-membership, 0=non co-membership (gene(row) * gene(col) matrix)<br/>
7th column: gene pairs selected for training (row, column, SL/non-SL {1,0}). The row and column index is started with 0<br/>
8th column: gene pairs selected for testing (row, column, SL/non-SL {1,0}). The row and column index is started with 0<br/>
9th column: the row counts of the input matrix<br/>

Citation
--------------------------------
If you find this useful for your research, we would be appreciated if you cite the following papers:<br/>

@article{liany2020predicting,<br/>
  title={Predicting synthetic lethal interactions using heterogeneous data sources},<br/>
  author={Liany, Herty and Jeyasekharan, Anand and Rajan, Vaibhav},<br/>
  journal={Bioinformatics},<br/>
  volume={36},<br/>
  number={7},<br/>
  pages={2209--2216},<br/>
  year={2020},<br/>
  publisher={Oxford University Press}<br/>
}<br/>
