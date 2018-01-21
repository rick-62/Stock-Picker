class Bank:

    def __init__(self, amount):
        self.current_value = amount
        self.original_value = amount

    @property
    def pct_increase(self):
        return 100 * \
               (self.current_value - self.original_value) / \
               self.original_value

    def __sub__(self, amount):
        """Amount can be directly subtracted from Bank object"""
        if self.current_value >= amount:
            self.current_value -= amount
            return True
        else:
            return False

    def __add__(self, amount):
        """Amount can be directly added to Bank object"""
        self.current_value += amount
        return True

    def __repr__(self):
        return str(self.current_value)
