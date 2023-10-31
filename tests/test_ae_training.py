import sys
from autoencoder_training_main import main

cases = {
    'basic': ["path_dataset=stock_data/portfolio_custom140_SHORT.csv", "n_epochs=1"]
}

# call main function from autoencoder_training_main.py and pass the arguments
for case in cases:
    print("Running case: ", case)
    sys.argv[1:] = cases[case]
    main()