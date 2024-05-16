"""
Given a list representing the price of a stock over a week, 
return the best day to buy and the best day to sell. 
Based on Best Time to Buy and Sell Stock II.
"""

prices = input().split()
for i in range(len(prices)):
    prices[i] = float(prices[i])

cur_hold, cur_not_hold = -float('inf'), 0
days = [None] * len(prices)

for i in range(len(prices)):
    # we don't have logic for taking back purchases
    stock_price = prices[i]
    prev_hold, prev_not_hold = cur_hold, cur_not_hold
    
    cur_hold = max(prev_hold, prev_not_hold - stock_price)
    if prev_hold < prev_not_hold - stock_price: # if it is more worth to buy
        days[i] = "buy"
        
    
    cur_not_hold = max(prev_not_hold, prev_hold + stock_price)
    if prev_not_hold < prev_hold + stock_price: # if it is more worth to sell
        days[i] = "sell" 
        
print(days)      

# one more O(n^2) pass to maximize profit  
    # O(n^2) okay since we have a max of 7
# keep an array of days to consider
# while values match last value added to array, add to days to consider
# if not, 
    # buy -> take lowest day
    # sell -> take highest day
    # all others set to null
is_buy = False
days_to_consider = []   

def buy_or_sell(is_buy, start, stop):
    if is_buy:
        min_price = min(prices[start:stop])
        # print(min_price)
        for j in range(start, stop):
            if prices[j] != min_price:
                days[j] = None
    else:    
        max_sell = max(prices[start:stop])
        for j in range(start, stop):
            if prices[j] != max_sell:
                days[j] = None

for i in range(len(days)):
    if (len(days_to_consider) > 0 and (days[i] == "buy") != is_buy):
        # print(days_to_consider)
        start, stop = days_to_consider[0], days_to_consider[-1] + 1
        buy_or_sell(is_buy, start, stop)
        days_to_consider = []
    days_to_consider.append(i) 
    is_buy = days[i] == 'buy'
    
if len(days_to_consider) > 0:
    buy_or_sell(is_buy, days_to_consider[0], days_to_consider[-1] + 1) 
            
    
print(cur_not_hold)
print(days)

