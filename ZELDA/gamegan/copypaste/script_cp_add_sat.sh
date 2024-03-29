#!/bin/zsh
setup_copy=('' '6 3,3,6,6' '58 2,2,5,8' '53 2,2,6,8' '58 6,3,9,7' '62 5,4,6,5')
setup_paste=('' '10 6,3,10,7' '6 10,2,13,8' '4 9,2,13,8' '3 4,4,5,6' '18 6,3,10,7')
setup_context=('' '11,12,18,27 6,3,10,7,6,3,10,7,6,3,10,7,6,3,10,7' '46,55,6,63 10,2,13,8,10,2,13,8,10,2,13,8,10,2,13,8'
 '7,16,17,26,55 9,2,13,8,9,2,13,8,9,2,13,8,9,2,13,8,9,2,13,8' '15,13,37,43 4,4,5,6,10,4,11,6,7,4,8,6,7,4,8,6'
 '10,11,12,54 6,3,10,7,6,3,10,7,6,3,10,7,6,3,10,7')

c_options=(16 64 256 512)
n_options=(500 1000 2500 5000)
l_options=(3 4)
d_options=(1)
lr_options=(0.0001 0.001 0.01 0.05)

for setup in {1..5}
do
    for c_opt in "${c_options[@]}"
    do
        for n_opt in "${n_options[@]}"
        do
            for l_opt in "${l_options[@]}"
            do
                for d_opt in "${d_options[@]}"
                do
                    for lr_opt in "${lr_options[@]}"
                    do
                        cd ..
                        printf '%s' "Constraint SAT: " >> copypaste/eval_output/setup${setup}/metrics_output_c${c_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.txt
                        python satisfaction_checker.py copypaste/eval_output/setup${setup}/rewritten_output_c${c_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.json >> copypaste/eval_output/setup${setup}/metrics_output_c${c_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.txt
                        cd ./copypaste
                        echo "Done with adding SAT $setup c ${c_opt} n ${n_opt} l ${l_opt} d ${d_opt} lr ${lr_opt}"
                    done
                done
            done
        done
    done
done