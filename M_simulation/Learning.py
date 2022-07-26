from src import module_No1_Preparing_data_for_training_FIM_RandL
from src import module_No2_data_learn_FIM2_Left
from src import module_No2_data_learn_FIM2_Right
from src import EDA_fim_data

if __name__ == "__main__":
    #Data Preparation
    module_No1_Preparing_data_for_training_FIM_RandL.advance_preparation_implementation()
    #Conducting model training and preservation
    module_No2_data_learn_FIM2_Left.Learning_and_Model_Saving()
    module_No2_data_learn_FIM2_Right.Learning_and_Model_Saving()
    #Preliminary Preparation Use of data to perform EDA analysis
    EDA_fim_data.EDA_all()