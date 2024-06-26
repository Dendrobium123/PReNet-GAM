#! bash 

# PReNet
python test_PReNet.py --logdir logs/Rain100L/PReNet_GAM_behindRES --save_path results/Rain100l/PReNet_GAM_behindRES_demo --data_path  datasets/test/Rain100L/rainy

# PReNet_r
python test_PReNet.py --logdir logs/Rain100L/PReNet6_r --save_path results/Rain100L/PReNet_r --data_path datasets/test/real

# PRN
python test_PRN.py --logdir logs/Rain100L/PRN6 --save_path results/Rain100L/PRN6 --data_path datasets/test/Rain100L/rainy

# PRN_r
python test_PRN_r.py --logdir logs/Rain100L/PRN6_r --save_path results/Rain100L/PRN_r --data_path datasets/test/Rain100L/rainy
