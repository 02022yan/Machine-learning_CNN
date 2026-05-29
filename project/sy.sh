#!/bin/bash
#SBATCH --exclude=sonmi1
# 定义变量
k_n=4
k_s=3
f_c=64
ir=0.0001
num_epochs=20000
###python generate_kernel.py $k_n $k_s
# 创建文件夹并复制脚本
for i in {1..5}
do
    folder_name="nopool_fc_${f_c}_kn_${k_n}_ks_${k_s}_ir_${ir}_epochs_${num_epochs}_i_${i}"
    mkdir -p "$folder_name"
    cp cnn_test.py "$folder_name/cnn_test.py"
    # 使用 sed 命令替换 Python 脚本中的变量
    sed -i "s/k_n = .*/k_n = $k_n/" "$folder_name/cnn_test.py"
    sed -i "s/k_s = .*/k_s = $k_s/" "$folder_name/cnn_test.py"
    sed -i "s/f_c = .*/f_c = $f_c/" "$folder_name/cnn_test.py"
    sed -i "s/ir = .*/ir = $ir/" "$folder_name/cnn_test.py"
    sed -i "s/num_epochs = .*/num_epochs = $num_epochs/" "$folder_name/cnn_test.py"

    # 在每个文件夹中运行脚本
    cd ./"$folder_name" 
    nohup srun python cnn_test.py & 
    cd -
done


# for ((i = 0; i < num_rows; i++)); do
    # if [ $i -eq 0 ]; then
        # row="[1"
    # else
        # row="["
        # for ((j = 0; j < 20; j++)); do
            # row="$row$(($RANDOM % 2))"
            # if [ $j -ne 19 ]; then
                # row="$row, "
            # fi
        # done
    # fi
    # row="$row]"
    # matrix="$matrix$row"
    # if [ $i -ne $(($num_rows - 1)) ]; then
        # matrix="$matrix, "
    # fi
# done
# matrix="$matrix]"
# Replace the line in 1.py with the new tensor assignment
# sed -i '' "s/self.conv1.weight.data = torch.tensor(\[.*\])/self.conv1.weight.data = torch.tensor($matrix, dtype=torch.float32)/g" "$folder_name/cnn_test.py"
