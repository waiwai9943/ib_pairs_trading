### Pair Trading Strategy with Kalman Filter and Interactive Broker API
This project implements a fully automatic pair trading strategy using the Kalman filter and the Interactive Broker API. The strategy is designed to analyze and trade pairs of financial instruments based on their cointegration relationship.

## Overview
The pair trading strategy aims to identify two financial instruments that are cointegrated, meaning they have a long-term relationship that can be exploited for trading opportunities. The strategy utilizes the Kalman filter, a mathematical algorithm, to estimate the spread between the two instruments and generate trading signals.

The key features of this project include:

1. Automatic Trading: The strategy handles the entire trading process, from identifying suitable pairs to placing orders, without manual intervention.
2. Cointegration Analysis: The strategy computes the cointegration between pairs of financial instruments to select the most appropriate pairs for trading.
3. Kalman Filter: The Kalman filter is used to estimate the spread between the selected pairs and generate trading signals based on the deviation from the mean.
4. Interactive Broker API Integration: The strategy connects to the Interactive Broker API to access real-time market data and execute trades.
## Functionality
The project provides the following functionality:

1. Pair Selection: If there are no existing positions in Interactive Broker, the strategy searches for pairs of financial instruments with the most suitable cointegration relationship. The pairs are sorted based on their p-values, indicating the strength of the cointegration.
2. Signal Generation: Once the pairs are selected, the strategy continuously updates the signals by calculating the spread between the instruments and comparing it to the mean. This information is used to generate trading signals.
3. Order Placement: When a trading signal is generated, the strategy automatically places the corresponding orders through the Interactive Broker API. This includes opening, closing, and adjusting positions as necessary.
4. Risk Management: The strategy includes risk management measures, such as position sizing and stop-loss orders, to ensure prudent risk control.
## Getting Started
To use this project, follow these steps:

1. Clone the repository: git clone [https://github.com/your-username/your-repo.git](https://github.com/waiwai9943/ib_pairs_trading.git
2. Install the required dependencies: pip install -r requirements.txt
3. Customize the strategy parameters, such as total capital and stop-loss levels, in the IB_pairs_trading_2.ipynb.py file.
4. Run the main script: IB_pairs_trading_2.ipynb.py

Please note that you will need an active Interactive Broker account with API access to fully utilize this project.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## Acknowledgments
We would like to acknowledge the following resources and libraries that have been instrumental in the development of this project:

Interactive Broker API: https://www.interactivebrokers.com/
Kalman Filter: https://en.wikipedia.org/wiki/Kalman_filter
Python: https://www.python.org/
Pandas: https://pandas.pydata.org/
NumPy: https://numpy.org/
Matplotlib: https://matplotlib.org/

## Contact
For any inquiries or further information, please contact Isaac WU at waiwai9943@gmail.com.

## Disclaimer
The content provided in this project, including the pair trading strategy, implementation details, and associated materials, is for informational purposes only. The author and contributors of this project do not guarantee the accuracy, completeness, or reliability of the information presented.

Trading and investing in financial markets involves substantial risk, and there is always the potential for financial loss. The strategies discussed or implemented in this project do not constitute financial advice or recommendations.

The author and contributors of this project shall not be held responsible for any losses, damages, or liabilities incurred as a result of using or relying on the information or strategies provided. It is essential that users exercise their own judgment and conduct thorough research and analysis before making any trading or investment decisions.

Users of this project are solely responsible for their actions and should seek professional advice from qualified financial advisors or brokers when considering engaging in trading activities.


