import sys
sys.path.append('/home/daniel/repositories/robobobo')

from autoencoder_training_main import main

cases = {
    'basic': ["path_dataset=stock_data/portfolio_custom140_SHORT.csv", "n_epochs=1", "target=full", "channels_out=40", "timeseries_out=10"],#, "load_checkpoint"]
}

# call main function from autoencoder_training_main.py and pass the arguments
for case in cases:
    print("Running case: ", case)
    sys.argv[1:] = cases[case]
    main()