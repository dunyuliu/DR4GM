#! /bin/bash

scenario_name="eqdyna.0001.B.100m"
output_dir=${scenario_name}
python eqdyna_converter_api.py --input_dir ../datasets/eqdyna/${scenario_name} --output_dir ${output_dir} --verbose
python npz_gm_processor.py --velocity_npz ${scenario_name}/eqdyna_velocities.npz --output_dir ${output_dir} #--chunk_size 1000
python gm_visualization.py --gm_npz ${scenario_name}/ground_motion_metrics.npz --output ${output_dir}
