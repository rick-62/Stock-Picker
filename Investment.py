class Investment:

    def __init__(self, fee, ticker_obj, price_per_share, num_shares, date):
        self.fee = fee
        self.ticker = ticker_obj
        self.date_bought = date
        self.price_paid = price_per_share
        self.current_price = price_per_share
        self.num_shares = num_shares
        self.cost = self.current_value
        self.status = True  # i.e. not sold

    def sell(self):
        """Initiated when investment is sold"""
        self.status = False
        return (self.current_value - self.fee,
                self.cost,
                self.current_price,
                self.price_paid,
                self.pct_change,
                self.num_shares,
                self.date_bought,
                self.fee,
                self.ticker.name)

    def update_value(self, new_price):
        """Updates current price/value of stock, being close of current day"""
        self.current_price = new_price

    @property
    def pct_change(self):
        """Returns percentage increase in value/price"""
        return (100 *
                (self.current_price - self.price_paid) /
                self.price_paid)

    def price_change(self, pct):
        """Returns minimum price given pct threshold"""
        return(self.price_paid * (1 + pct / 100))

    @property
    def current_value(self):
        """Returns current total value of investment, inc fees"""
        return (self.current_price / 100) * self.num_shares - self.fee

    def capped_value(self, pct):
        """Returns current value capped by percent change"""
        return (self.price_change(pct) / 100) * self.num_shares - self.fee

    def __repr__(self):
        return self.ticker.name
