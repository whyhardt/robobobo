import sys
from forecast_training_main import main

cases = {
    'basic': [
        "path_dataset=stock_data/portfolio_custom140_SHORT.csv", 
        "n_epochs=1", 
        "forecast_length=1", 
        ],
    'load_checkpoint': [
        "path_dataset=stock_data/portfolio_custom140_SHORT.csv", 
        "n_epochs=1", 
        "forecast_length=1",
        "load_checkpoint",
        ],
    'ae': [
        "path_dataset=stock_data/portfolio_custom140_SHORT.csv", 
        "n_epochs=1", 
        "forecast_length=1", 
        "path_autoencoder=../trained_ae/robobobo_ae20_p140normrange.pt"
        ],
    'load_checkpoint_ae': [
        "path_dataset=stock_data/portfolio_custom140_SHORT.csv",
        "load_checkpoint",
        "n_epochs=1", 
        "forecast_length=1", 
        "path_autoencoder=../trained_ae/robobobo_ae20_p140normrange.pt"
        ],
}

# call main function from autoencoder_training_main.py and pass the arguments
for case in cases:
    print("Running case: ", case)
    sys.argv[1:] = cases[case]
    main()