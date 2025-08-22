#!/bin/bash

scenario_name="waveqlab3d.0001.A"
output_dir=${scenario_name}
python waveqlab3d_converter_api.py --input_dir ../datasets/waveqlab3d --output_dir ${output_dir} --verbose
python npz_gm_processor.py --velocity_npz ${scenario_name}/waveqlab3d_velocities.npz --output_dir ${output_dir} #--chunk_size 10000
python gm_visualization.py --gm_npz ${scenario_name}/ground_motion_metrics.npz --output ${output_dir}