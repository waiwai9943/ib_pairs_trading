Pair Trading Strategy with Kalman Filter and Interactive Broker API
This project implements a fully automatic pair trading strategy using the Kalman filter and the Interactive Broker API. The strategy is designed to analyze and trade pairs of financial instruments based on their cointegration relationship.

Overview
The pair trading strategy aims to identify two financial instruments that are cointegrated, meaning they have a long-term relationship that can be exploited for trading opportunities. The strategy utilizes the Kalman filter, a mathematical algorithm, to estimate the spread between the two instruments and generate trading signals.

The key features of this project include:

Automatic Trading: The strategy handles the entire trading process, from identifying suitable pairs to placing orders, without manual intervention.
Cointegration Analysis: The strategy computes the cointegration between pairs of financial instruments to select the most appropriate pairs for trading.
Kalman Filter: The Kalman filter is used to estimate the spread between the selected pairs and generate trading signals based on the deviation from the mean.
Interactive Broker API Integration: The strategy connects to the Interactive Broker API to access real-time market data and execute trades.
Functionality
The project provides the following functionality:

Pair Selection: If there are no existing positions in Interactive Broker, the strategy searches for pairs of financial instruments with the most suitable cointegration relationship. The pairs are sorted based on their p-values, indicating the strength of the cointegration.
Signal Generation: Once the pairs are selected, the strategy continuously updates the signals by calculating the spread between the instruments and comparing it to the mean. This information is used to generate trading signals.
Order Placement: When a trading signal is generated, the strategy automatically places the corresponding orders through the Interactive Broker API. This includes opening, closing, and adjusting positions as necessary.
Risk Management: The strategy includes risk management measures, such as position sizing and stop-loss orders, to ensure prudent risk control.
Getting Started
To use this project, follow these steps:

Clone the repository: git clone https://github.com/your-username/your-repo.git
Install the required dependencies: pip install -r requirements.txt
Configure the Interactive Broker API credentials in the config.py file.
Customize the strategy parameters, such as position sizing and stop-loss levels, in the strategy.py file.
Run the main script: python main.py
Please note that you will need an active Interactive Broker account with API access to fully utilize this project.

Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License.

Acknowledgements
We would like to acknowledge the following resources and libraries that have been instrumental in the development of this project:

Interactive Broker API: https://www.interactivebrokers.com/
Kalman Filter: https://en.wikipedia.org/wiki/Kalman_filter
Python: https://www.python.org/
Pandas: https://pandas.pydata.org/
NumPy: https://numpy.org/
Matplotlib: https://matplotlib.org/
Contact
For any inquiries or further information, please contact Isaac WU at waiwai9943@gmail.com.

Happy Trading!
