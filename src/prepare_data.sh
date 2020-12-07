python generate_graph.py ../data/imagelist.txt ../data/refer360images ../data/graph_data_top100
python generate_grid.py ../data/imagelist.txt ../data/refer360images ../data/grid_data_30degrees 30
python generate_moves.py ../data/grount_truth_moves
python generate_object_dictionaries.py ../data/imagelist.txt ../data/graph_data_top100  ../data/vg_object_dictionaries.all.json all
python generate_object_dictionaries.py ../data/imagelist.txt ../data/graph_data_top100  ../data/vg_object_dictionaries.top100.json top100
python generate_object_dictionaries.py ../data/imagelist.txt ../data/graph_data_top100  ../data/vg_object_dictionaries.top50.json top50
python dump_data.py  ../data/continuous_grounding all
python dump_data.py  ../data/graph_grounding all graph_grounding ../data/graph_data_top100
python dump_data.py  ../data/fov_pretraining all fov_pretraining ../data/grount_truth_moves/ ../data/graph_data_top100 ../data/vg_object_dictionaries.top100.json
python dump_data.py  ../data/grid_fov_pretraining all grid_fov_pretraining ../data/grid_data_30degrees/ ../data/graph_data_top100 ../data/vg_object_dictionaries.top100.json
