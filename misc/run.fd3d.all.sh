#!/bin/bash

scenario_name="fd3d.0001.A"
output_dir=${scenario_name}
python fd3d_converter_api.py --input_dir ../datasets/fd3d/nucl_cent --output_dir ${output_dir} --verbose
python station_subset_selector.py --input_npz ${scenario_name}/fd3d_velocities.npz --output_npz ${scenario_name}/subset_step10.npz --station_step 10
python npz_gm_processor.py --velocity_npz ${scenario_name}/subset_step10.npz --output_dir ${output_dir} #--chunk_size 10000
python gm_visualization.py --gm_npz ${scenario_name}/ground_motion_metrics.npz --output ${output_dir}
