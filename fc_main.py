import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from copy import deepcopy

from helpers.init_ae import init_ae
from nn_architecture.forecasting_networks import TradeNet
from environment import Environment


def main(
    load_checkpoint=False,
    file_checkpoint='trained_fc/checkpoint.pt',
    file_data='stock_data/portfolio_custom129_2002_2023_normrange.csv',
    file_ae='trained_ae/ae129.pt',
    num_epochs=1e1, 
    checkpoint_interval=1e1,
    num_random_actions=5e2,
    batch_size=32,
    learning_rate=3e-4,
    hidden_dim=32,
    num_layers=3,
    sequence_length=10,
    dropout=0.,
    portfolio_input=False,
    ):
    """main file for training and testing a forecasting network"""
    
    cfg = {
        # general parameters
        'load_checkpoint': load_checkpoint,
        'file_checkpoint': file_checkpoint,
        'file_data': file_data,
        'file_predictor': [None, None],  # ['trained_gan/real_gan_1k.pt', 'trained_gan/mvgavg_gan_10k.pt',],
        'file_ae': file_ae,
        'checkpoint_interval': checkpoint_interval,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        # training parameters
        'num_epochs': num_epochs,
        'num_random_actions': num_random_actions,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'train_test_split': 0.8,
        'replay_buffer_size': int(1e4),
        'parameter_update_interval': 50,
        'sequence_length': sequence_length,
        'dropout': dropout,

        # network parameters
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_layers_sub': 4,
        'init_w': None,
        'portfolio_input': portfolio_input,

        # environment
        'time_limit': 365,
        'cash_init': 10000,
        'commission': .001,
        'reward_scaling': 1e-2,
    }
    
    training_data = pd.read_csv(cfg['file_data'], index_col=0, header=0)
    portfolio_names = training_data.columns
    training_data = training_data.to_numpy(dtype=np.float32)
    test_data = np.expand_dims(training_data[int(cfg['train_test_split']*len(training_data)):], axis=0)
    training_data = training_data[:int(cfg['train_test_split']*len(training_data))]
    # cut training_data into sequences of lenght cfg['sequence_length']
    training_data = np.stack([training_data[i:i+(cfg['sequence_length']+1)] for i in range(len(training_data)-(cfg['sequence_length']+1))], axis=0)
    training_data = torch.tensor(training_data, dtype=torch.float32, device=cfg['device'])
    
    # get test data
    test_data_numpy = test_data
    test_data = torch.tensor(test_data, dtype=torch.float32, device=cfg['device'])
    
    # get autoencoder as feature extractor for 
    if cfg['file_ae'] is not None and cfg['file_ae'] != '':
        state_dict = torch.load(cfg['file_ae'], map_location=torch.device('cpu'))
        encoder = init_ae(**state_dict['configuration'], sequence_length=16)
        encoder.load_state_dict(state_dict['model'])
        encoder.to(cfg['device'])
        encoder.eval()
    else:
        encoder = None
    
    # set robobobo
    robobobo = TradeNet(
        encoder.output_dim if encoder is not None else training_data.shape[-1], 
        cfg['hidden_dim'], 
        cfg['sequence_length'], 
        encoder.input_dim if encoder is not None else training_data.shape[-1], 
        cfg['num_layers'], 
        cfg['dropout'],
        n_portfolio=training_data.shape[-1] if cfg['portfolio_input'] else 0
        ).to(cfg['device'])
    
    # set loss function and equity calculation function
    def trade_loss(equity_trade):
        # return torch.mean(torch.sum(equity_keep - equity_trade, dim=-1))
        return - torch.mean(equity_trade)
    
    # set optimizer
    optim = torch.optim.Adam(robobobo.parameters(), lr=cfg['learning_rate'])
    
    # load checkpoint
    if cfg['load_checkpoint']:
        checkpoint = torch.load(cfg['file_checkpoint'])
        robobobo.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
    
    # train loop
    try:
        for epoch in range(int(cfg['num_epochs'])):
            epoch_loss = []
            # shuffle training_data
            training_data = training_data[torch.randperm(training_data.shape[0])]
            for i in range(0, len(training_data), cfg['batch_size']):
                # get batch
                batch = training_data[i:i+cfg['batch_size']]
                
                # zero gradients
                optim.zero_grad()
                # forward pass
                if encoder is not None:
                    with torch.no_grad():
                        encoded_batch = encoder.encode(batch[:, :-1, :])
                else:
                    encoded_batch = batch[:, :-1, :]
                if robobobo.portfolio:
                    # set a random portfolio of size (batch_size, 1, n_portfolio) with n random integers between 0 and 100
                    rnd_portfolio = torch.randint(0, 100, (batch.shape[0], 1, training_data.shape[-1]), device=cfg['device'], dtype=torch.int)
                    n_zeros = torch.randint(training_data.shape[-1]//2, training_data.shape[-1], (batch.shape[0],), device=cfg['device'], dtype=torch.int)
                    for i in range(batch.shape[0]):
                        index_zeros = torch.randint(0, training_data.shape[-1], (n_zeros[i],), device=cfg['device'], dtype=torch.int)
                        rnd_portfolio[i, 0, index_zeros] = 0
                    orders_buy, orders_sell = robobobo(encoded_batch, rnd_portfolio)
                    rnd_portfolio = rnd_portfolio.squeeze(1)
                else:
                    # continue without information about portfolio
                    orders_buy, orders_sell = robobobo(encoded_batch, None)
                orders_sell = orders_sell * -1
                
                # calculate trade effectivity
                # V1
                diff_stocks = batch[:, -1, :] - batch[:, -2, :]
                trade_buy = torch.sum(torch.prod(torch.stack((orders_buy, diff_stocks), dim=1), dim=1), dim=-1)
                trade_sell = torch.sum(torch.prod(torch.stack((orders_sell, diff_stocks), dim=1), dim=1), dim=-1)
                effectivity_trade = trade_buy - trade_sell
                # V2
                # trade_sell = torch.sum((orders_sell * rnd_portfolio).int() * batch[:, -2, :], dim=-1)
                # trade_buy = orders_buy * trade_sell.unsqueeze(-1)
                # effectivity_trade = trade_buy * (batch[:, -1, :] - batch[:, -2, :]) * cfg['reward_scaling']
                
                # calculate loss
                loss = trade_loss(effectivity_trade)
                epoch_loss.append(loss.item())
                
                # backward pass
                loss.backward()
                optim.step()
            
            # print epoch loss
            print(f'Epoch {epoch+1} loss: {np.mean(epoch_loss)}')
    
        if cfg['num_epochs'] != 0:
            # save checkpoint
            checkpoint = {
                'configuration': cfg,
                'model': robobobo.state_dict(),
                'optimizer': optim.state_dict(),
            }
            torch.save(checkpoint, cfg['file_checkpoint'])
    except KeyboardInterrupt:
        print('Training interrupted. Saving checkpoint and continue with testing the trained model.')
        # save checkpoint
        checkpoint = {
            'configuration': cfg,
            'model': robobobo.state_dict(),
            'optimizer': optim.state_dict(),
        }
        torch.save(checkpoint, cfg['file_checkpoint'])
        
    # test trained robobobo
    
    # ----------------------
    # Test with environment
    # ----------------------
    
    # # initialize environment
    # env = Environment(
    #     stock_data=test_data_numpy[0],
    #     portfolio_names=portfolio_names,
    #     cash=cfg['cash_init'],
    #     observation_length=cfg['sequence_length'],
    #     test=True,
    #     recurrent=True,
    #     commission_buy=0,
    #     commission_sell=0,
    #     # reward_scaling=cfg['reward_scaling'],
    # )
    
    # for t in range(cfg['sequence_length'], test_data.shape[1]-2):
    #     # get stocks
    #     stocks_history = test_data[0, t-cfg['sequence_length']:t, :].unsqueeze(0)
    #     stocks_next = test_data[0, t, :]
        
    #     # get encoded stocks
    #     with torch.no_grad():
    #         orders_buy, orders_sell = robobobo(encoder.encode(stocks_history))
            
    #         # get orders
    #         obs, _ , _, _, _ = env.step((orders_buy+orders_sell).cpu().numpy()[0])
    #         stocks_history = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32, device=cfg['device'])
    
    # ----------------------
    # Test without environment
    # ----------------------
    
    # initialize robobobo
    cash = torch.tensor(cfg['cash_init'], dtype=torch.float32, device=cfg['device'])
    portfolio = torch.zeros((1, test_data.shape[-1]), dtype=torch.float32, device=cfg['device'])
    
    total_equity_array = []
    equity_avg_array = []
    cash_array = []
    portfolio_array = []
    
    not_masked = test_data[0, cfg['sequence_length'], :] != 0
    avg_portfolio = torch.zeros((1, test_data.shape[-1]), dtype=torch.float32, device=cfg['device'])
    avg_portfolio[0, not_masked] = torch.ones_like(portfolio)[0, not_masked] * (cash/portfolio[0, not_masked].shape[-1]) / test_data[0, cfg['sequence_length'], not_masked]
    
    # start loop
    for t in range(cfg['sequence_length'], test_data.shape[1]-2):
        # get stocks
        stocks_history = test_data[0, t-cfg['sequence_length']:t, :].unsqueeze(0)
        stocks_next = test_data[0, t, :]
        
        buy_amounts = torch.zeros((1, test_data.shape[-1]), device=cfg['device'])
        sell_amounts = torch.zeros((1, test_data.shape[-1]), device=cfg['device'])
        
        not_masked = stocks_history[0, -1, :] != 0
        
        # get encoded stocks
        with torch.no_grad():
            if encoder is not None:
                encoded_stocks = encoder.encode(stocks_history)
            else:
                encoded_stocks = stocks_history
            # get orders
            orders_buy, orders_sell = robobobo(encoded_stocks, portfolio if cfg['portfolio_input'] else None)
            orders_sell = orders_sell * -1
            
            # selling orders
            sell_amounts[0, not_masked] = orders_sell[0, not_masked] * portfolio[0, not_masked]
            sell = sell_amounts[0, not_masked] * stocks_history[0, -1, not_masked]
            # update cash
            cash = cash + torch.sum(sell, dim=-1)
            # update portfolio
            portfolio = portfolio - sell_amounts
            cash_after_sell = deepcopy(cash.item())
            
            # buying orders
            invest = orders_buy * cash
            buy_amounts[0, not_masked] = invest[0, not_masked] / stocks_history[0, -1, not_masked]
            # update cash
            cash = cash - torch.sum(invest, dim=-1)
            # update portfolio
            portfolio = portfolio + buy_amounts
            
            # check if any portfolio entry is nan
            # if torch.any(portfolio < 0):
            #     print(f'Some entry in portfolio is negative at time {t}')
            #     break
            # if torch.any(torch.isnan(portfolio)):
            #     print(f'Nan in portfolio at time {t}')
            #     break
            
            # calculate total equity
            total_equity = cash + torch.sum(portfolio * stocks_next, dim=-1)
            total_equity_array.append(total_equity.item())
            cash_array.append(cash_after_sell)
            equity_avg_array.append(torch.sum(avg_portfolio * stocks_next, dim=-1).item())
            portfolio_array.append(deepcopy(portfolio.cpu().numpy()))
    
    # plot total equity
    fig, axs = plt.subplots(3)
    axs[0].plot(total_equity_array, label='portfolio')
    axs[0].plot(equity_avg_array, linestyle='--', label='benchmark')
    axs[0].axes.get_xaxis().set_visible(False)
    axs[0].set_title('Total Equity')
    # set legend for axs[0]
    axs[0].legend()
    # plot cash after sale orders over time
    axs[1].plot(cash_array)
    axs[1].set_title('Cash')
    axs[1].axes.get_xaxis().set_visible(False)
    # plot portfolio over time as a matrix where each row is a stock and each column is a time step
    portfolio_array = np.concatenate(portfolio_array, axis=0)
    axs[2].imshow(portfolio_array.T, aspect='auto')
    plt.show()
    
    
if __name__ == '__main__':
    main(
        load_checkpoint=False,
        file_ae=None,
        num_epochs=1,
        hidden_dim=256,
        portfolio_input=True,
        learning_rate=1e-5,
        )