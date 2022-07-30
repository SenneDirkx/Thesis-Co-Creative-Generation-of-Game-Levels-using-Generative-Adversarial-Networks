#!/bin/zsh
setup_copy=('' '6 3,3,6,6' '58 2,2,5,8' '53 2,2,6,8' '3 4,0,5,10' '62 5,4,6,5')
setup_paste=('' '10 6,3,10,7' '6 10,2,13,8' '4 9,2,13,8' '47 7,0,8,10' '18 6,3,10,7')
setup_context=('' '11,12,18,27 6,3,10,7,6,3,10,7,6,3,10,7,6,3,10,7' '46,55,6,63 10,2,13,8,10,2,13,8,10,2,13,8,10,2,13,8'
 '7,16,17,26,55 9,2,13,8,9,2,13,8,9,2,13,8,9,2,13,8,9,2,13,8' '63,38,34,20 7,0,8,10,7,0,8,10,7,0,8,10,0,4,4,10'
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
                        python interact.py ./ZeldaDungeon02_5000_10Z.pth 10 4 11 16 ${setup_paste[$setup]} ${setup_context[$setup]} ${v_opt} ${n_opt} $c_opt latent_input.txt ${setup_edit[$setup]} ${l_opt} ./eval_output_custom_local/setup${setup}/rewritten_custom_generator_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.pth ${d_opt} ${lr_opt} > ./eval_output_custom_local/setup${setup}/metrics_output_c${c_opt}_n${n_opt}_v${v_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.txt
                        #touch copypaste/eval_output/setup${setup}/metrics_output_c${c_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.txt
                        python main.py ./eval_output_custom_local/setup${setup}/rewritten_custom_generator_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.pth 10 4 11 16 ./latent_input.txt ./eval_output_custom_local/setup${setup}/rewritten_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.json
                        python json_to_image.py ./eval_output_custom_local/setup${setup}/rewritten_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.json ./eval_output_custom_local/setup${setup}/rewritten_image_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.png
                        printf '%s' "Diversity Dice: " >> ./eval_output_custom_local/setup${setup}/metrics_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.txt
                        python main.py ./eval_output_custom_local/setup${setup}/rewritten_custom_generator_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.pth 10 4 11 16 ./diversity_input.txt ./eval_output_custom_local/setup${setup}/rewritten_distance_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.json
                        python diversity.py ./eval_output_custom_local/setup${setup}/rewritten_distance_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.json dice >> ./eval_output_custom_local/setup${setup}/metrics_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.txt
                        printf '%s' "Diversity LPIPS: " >> ./eval_output_custom_local/setup${setup}/metrics_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.txt
                        python diversity.py ./eval_output_custom_local/setup${setup}/rewritten_distance_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.json lpips >> ./eval_output_custom_local/setup${setup}/metrics_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.txt
                        printf '%s' "Constraint SAT: " >> ./eval_output_custom_local/setup${setup}/metrics_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.txt
                        python satisfaction_checker.py ./eval_output_custom_local/setup${setup}/rewritten_distance_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.json >> ./eval_output_custom_local/setup${setup}/metrics_output_c${c_opt}_v${v_opt}__n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.txt
                        rm ./eval_output_custom_local/setup${setup}/rewritten_distance_output_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.json
                        rm ./eval_output_custom_local/setup${setup}/rewritten_custom_generator_c${c_opt}_v${v_opt}_n${n_opt}_l${l_opt}_d${d_opt}_lr${lr_opt}.pth
                        echo "Done with setup $setup c ${c_opt} v ${v_opt} n ${n_opt} l ${l_opt} d ${d_opt} lr ${lr_opt}"
                    done
                done
            done
        done
    done
done