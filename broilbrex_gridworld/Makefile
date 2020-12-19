OBJS = build/birl_test.o build/feature_test.o build/mdp_test.o
CC = g++ -std=c++11
DEBUG = -g -O3 -fopenmp 
CFLAGS = -Wall -c $(DEBUG) 
LFLAGS = -Wall $(DEBUG)


all: birl_test feature_test mdp_test feature_birl_test conf_bound_test grid_experiment conf_bound_feature_test grid_l1_experiment fcount_example driving_example driving_birl_example experiment1 experiment2 experiment3 experiment4 experiment5 experiment7 experiment8 driving_experiment1 driving_birl_sandbox experiment9 experiment10 poison_test teaching1 enough1 enough2 enough_toy improvement_toy gen_feasible_r cakmak_test

gridworld_basic_exp: build/gridWorldBasicExperiment.o
	$(CC) $(LFLAGS) build/gridWorldBasicExperiment.o  -o gridworld_basic_exp

brex_gridworld_basic_exp_optsubopt: build/brex_gridWorldBasicExperimentOptSubopt.o
	$(CC) $(LFLAGS) build/brex_gridWorldBasicExperimentOptSubopt.o  -o brex_gridworld_basic_exp_optsubopt


brex_gridworld_basic_exp_goodbad: build/brex_gridWorldBasicExperimentGoodBad.o
	$(CC) $(LFLAGS) build/brex_gridWorldBasicExperimentGoodBad.o  -o brex_gridworld_basic_exp_goodbad

brex_gridworld_basic_exp_ranked: build/brex_gridWorldBasicExperimentRanked.o
	$(CC) $(LFLAGS) build/brex_gridWorldBasicExperimentRanked.o  -o brex_gridworld_basic_exp
	

brex_gridworld_basic_exp_optnoise: build/brex_gridWorldBasicExperimentOptNoise.o
	$(CC) $(LFLAGS) build/brex_gridWorldBasicExperimentOptNoise.o  -o brex_gridworld_basic_exp_optnoise

brex_test: build/brex_test.o
	$(CC) $(LFLAGS) build/brex_test.o  -o brex_test

brex_toy_test: build/brex_toy_test.o
	$(CC) $(LFLAGS) build/brex_toy_test.o  -o brex_toy_test

brex_toy_test_noiserankings: build/brex_toy_test_noiserankings.o
	$(CC) $(LFLAGS) build/brex_toy_test_noiserankings.o  -o brex_toy_test_noiserankings

build/gridWorldBasicExperiment.o: src/gridWorldBasicExperiment.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridWorldBasicExperiment.cpp -o build/gridWorldBasicExperiment.o

build/brex_gridWorldBasicExperimentOptSubopt.o: src/brex_gridWorldBasicExperimentOptSubopt.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/brex_gridWorldBasicExperimentOptSubopt.cpp -o build/brex_gridWorldBasicExperimentOptSubopt.o

build/brex_gridWorldBasicExperimentOptNoise.o: src/brex_gridWorldBasicExperimentOptNoise.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/brex_gridWorldBasicExperimentOptNoise.cpp -o build/brex_gridWorldBasicExperimentOptNoise.o

build/brex_gridWorldBasicExperimentGoodBad.o: src/brex_gridWorldBasicExperimentGoodBad.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp include/feature_brex.hpp
	$(CC) $(CFLAGS) src/brex_gridWorldBasicExperimentGoodBad.cpp -o build/brex_gridWorldBasicExperimentGoodBad.o

build/brex_gridWorldBasicExperimentRanked.o: src/brex_gridWorldBasicExperimentRanked.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp include/feature_brex.hpp
	$(CC) $(CFLAGS) src/brex_gridWorldBasicExperimentRanked.cpp -o build/brex_gridWorldBasicExperimentRanked.o

build/brex_test.o: src/brex_test.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_brex.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/brex_test.cpp -o build/brex_test.o

build/brex_toy_test.o: src/brex_toy_test.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_brex.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/brex_toy_test.cpp -o build/brex_toy_test.o
	
build/brex_toy_test_noiserankings.o: src/brex_toy_test_noiserankings.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_brex.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/brex_toy_test_noiserankings.cpp -o build/brex_toy_test_noiserankings.o


clean:
	\rm build/*.o src/*~ gridworld_basic_exp brex_gridworld_basic_exp brex_gridworld_basic_exp_goodbad brex_gridworld_basic_exp_optnoise brex_gridworld_basic_exp_optsubopt brex_test brex_toy_test brex_toy_test_noiserankings
	    
