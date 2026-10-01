"""
Simulation to determine whether straights or flushes are easier to get in the game Balatro

A starting had has 8 cards, if you can discard and then draw up to 5 cards, how common is it to complete a flush vs straight
"""
import eval7, pprint
import numpy as np
from scipy.special import comb, factorial
from scipy.stats import hypergeom
import matplotlib.pyplot as plt

import time
from timer import time_methods

#@time_methods
class PokerHands:

    # References
    RANKS = ('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A')
    SUITS = ('c', 'd', 'h', 's')
    HAND_SIZE = 5
    

    def __init__(self, hand, deck, starting_hand_size=8, verbose = True):

        self.starting_hand_size=starting_hand_size
        self.deck=deck
        self.hand=[]
        self.obs_suits = {} # history of all observed suits
        self.obs_ranks = {} # history of all observed ranks

        self.min_suit_count = self.starting_hand_size // 4 # absolute minimum num of suits that is possible to have in hand for intended flush suit
        self.update(hand)
        

        self.straight_references = [['A','2', '3', '4', '5']] + [self.RANKS[i:i+5] for i in range(9)]

        # Settings
        self.verbose = verbose

    def get_counts(self, x_list, count_dict={}):
        """Get count of elements in a list"""
        for x in x_list:
            count_dict[x] = 1 + count_dict.get(x,0)
        return count_dict

    def get_suit_count(self, hand, count_dict={}):
        """Counts the number of suits, and updates if adding to an existing count dict"""
        current_suit_count = self.get_counts(
            [self.SUITS[h.suit] for h in hand], 
            count_dict)
        return current_suit_count

    def get_rank_count(self, hand, count_dict={}):
        """Counts the number of ranks, and updates if adding to an existing count dict"""
        current_rank_count = self.get_counts(
            [self.RANKS[h.rank] for h in hand], 
            count_dict)
        return current_rank_count

    def update(self, hand):
        """Update hand, historical observed suits, and observed historical ranks"""
        self.hand += hand 
        self.obs_suits = self.get_suit_count(hand, self.obs_suits)
        self.obs_ranks = self.get_rank_count(hand, self.obs_ranks)
         
    def is_flush(self):
        """Detects if hand has a flush"""
        evaluated_hand = eval7.evaluate(self.hand)
        is_flush = (eval7.handtype(evaluated_hand) == "Flush") or (eval7.handtype(evaluated_hand) == "Straight Flush")
        return is_flush
    
    def is_straight(self):
        """Detects if hand has a straight"""
        evaluated_hand = eval7.evaluate(self.hand)
        is_straight = (eval7.handtype(evaluated_hand) == "Straight") or (eval7.handtype(evaluated_hand) == "Straight Flush")
        return is_straight

    def simulate_one_round(self, hand_type, hand_eval, stop_cond):
        """
        Simulates one round to play 1 hand
        
        Args:
            hand_type (str): string signalling which hand type to aim for
            hand_eval (func): method for picking which type of hand to play
            stop_cond (func): method for end
        
        Returns:
            num_discards (int): Number of discards it took to achieve hand. np.nan if not achieved
        """

        # Preparing methods
        if hand_type == "flush":
            current_hand = self.obs_suits
            stop_cond = self.is_flush
            get_count = lambda hand: self.get_suit_count(hand, count_dict={})
        elif hand_type == "straight":
            current_hand = self.obs_ranks
            stop_cond = self.is_straight
            get_count = lambda hand: self.get_rank_count(hand, count_dict={})
        else:
            self.raise_hand_type_error(hand_type=hand_type)

        # Executing loop
        num_discards = 0
        while not stop_cond():
            # Picking which poker hand to aim for
            poker_eval = hand_eval(current_hand, hand_type) # output keys are sorted
            max_poker_val = max(poker_eval.values())
            max_poker_hands = [poker for poker, val in poker_eval.items() if val == max_poker_val]
            if hand_type == "flush":
                target_poker_hand = np.random.choice(max_poker_hands).item() # randomly pick
            elif hand_type == "straight":
                target_poker_hand = max_poker_hands[-1] # pick highest viable straight, assumes sorted

            # Picking which cards to keep
            hand_to_keep = self.get_hand_to_keep(hand_type, target_poker_hand)

            # Executing discard
            num_cards_left_in_deck = len(self.deck)
            cards_to_discard = self.starting_hand_size - len(self.hand)
            card_to_draw = min(cards_to_discard, num_cards_left_in_deck)
            if num_cards_left_in_deck <= 0:
                if self.verbose:
                    print("Ran out of cards to draw")
                num_discards = np.nan
                return num_discards
            new_cards = self.deck.deal(card_to_draw)
            
            # Updating for next loop
            self.hand = hand_to_keep
            self.update(new_cards) # add new cards hand and dict of observed cards
            current_hand = get_count(self.hand)
            num_discards += 1

        if self.verbose:
            print(f"It took {num_discards} discards to achieve a straight")
        return num_discards

    def get_hand_to_keep(self, hand_type, target_poker_hand):
        """Preparing cards to keep in hand to throw the rest away"""

        # Select cards to keep
        if hand_type == "flush":
            hand_to_keep = [
                card for card in self.hand
                if self.SUITS[card.suit]==target_poker_hand
                ] # filter to keep only target suit
        elif hand_type == "straight":
            hand_to_keep = [
                card for card in self.hand
                if self.RANKS[card.rank] in target_poker_hand.split(",")
                ] # filter to keep only target suit
        else:
            self.raise_hand_type_error(hand_type=hand_type)

        if hand_to_keep == []:
            raise ValueError("Selected poker hand with no viable cards in hand")
        

        # Remove duplicates (only applies to straights)
        hand_to_keep_has_duplicates = False
        if hand_type == "straight":
            current_ranks = self.get_rank_count(hand_to_keep, count_dict={}) # get rank count to check for duplicates
            max_rank_count = max(current_ranks.values())
            hand_to_keep_has_duplicates = max_rank_count > 1
        
        if hand_type == "straight" and hand_to_keep_has_duplicates:
            # for each duplicate rank, pick a random card
            duplicate_ranks_in_hand_to_keep = [rank for rank, count in current_ranks.items() if count == max_rank_count] # identified duplicate ranks
            duplicate_rank_to_keep = [
                next(card for card in hand_to_keep if self.RANKS[card.rank] == rank) # picking one card of a given rank 
                for rank in duplicate_ranks_in_hand_to_keep
                ]
            
            # recompose hand to keep as unique ranks + deduplicated ranks
            hand_to_keep = [card for card in hand_to_keep if self.RANKS[card.rank] not in duplicate_ranks_in_hand_to_keep] + duplicate_rank_to_keep

        # Add back cards so discards are at most hand_size (5)
        num_cards_to_discard = self.starting_hand_size - len(hand_to_keep)
        discarding_too_many_cards = num_cards_to_discard > self.HAND_SIZE
        if discarding_too_many_cards:
            # need to add a random card back
            num_cards_to_add_back = num_cards_to_discard - self.HAND_SIZE
            cards_not_in_hand = iter([card for card in self.hand if card not in hand_to_keep])
            random_cards = [next(cards_not_in_hand, None) for _ in range(num_cards_to_add_back)]
            hand_to_keep += random_cards
        
        return hand_to_keep

    def get_greedy_flush(self, current_suits, hand_type):
        """Pick flush based on which suit is the most complete"""
        if hand_type != "flush":
            raise_wrong_hand_type_error("get_greedy_flush", hand_type)
        return current_suits

    def get_greedy_straight(self, current_ranks, hand_type):
        if hand_type != "straight":
            raise_wrong_hand_type_error("get_greedy_straight", hand_type)
        hand_rank_set = set(current_ranks.keys())
        straight_hits = {",".join(straight): len(set(straight).intersection(hand_rank_set)) for straight in self.straight_references}
        return straight_hits

    def get_likely_flush(self, current_suits, hand_type):
        if hand_type != "flush":
            raise_wrong_hand_type_error("get_likely_flush", hand_type)
        return self.get_likelihood(current_suits, hand_type="flush")

    def get_likely_straight(self, current_ranks, hand_type):
        if hand_type != "straight":
            raise_wrong_hand_type_error("get_likely_straight", hand_type)
        return self.get_likelihood(current_ranks, hand_type="straight")

    # def simulate_greedy_flush(self):
    #     """Simulate one round with the goal of getting a flush using the greedy method"""
    #     # Method only goes for the most abundant suit in hand, not what is most likely to be drawn

    #     num_discards = 0
    #     current_suits = self.obs_suits
    #     while not self.is_flush(current_suits):
    #         # Deciding which suit to keep
    #         max_count = max(current_suits.values())
    #         most_common_suits = [suit for suit, count in current_suits.items() if count == max_count]

    #         # Removing cards and then updating
    #         suit_to_keep = np.random.choice(most_common_suits).item() # randomly pick
    #         hand_to_keep = [
    #             card for card in self.hand
    #             if self.SUITS[card.suit]==suit_to_keep
    #             ] # filter to keep only target suit
    #         if len(hand_to_keep) <= self.min_suit_count:
    #             # if most common suit has 2 counts, then we need to keep 1 more card to satisfy hand size
    #             # only happens if suit count is 2-2-2-2 and hand_size is 8
    #             random_card = next(card for card in self.hand if self.SUITS[card.suit]!=suit_to_keep)
    #             hand_to_keep += [random_card]
    #         self.hand = hand_to_keep

    #         cards_to_discard = self.starting_hand_size - len(self.hand)
    #         discarded_more_than_hand_size = cards_to_discard > self.HAND_SIZE
    #         if discarded_more_than_hand_size:
    #             raise ValueError(f"Error: Discarded more than {self.HAND_SIZE} cards, but ended up discarding {cards_to_discard} cards")
            
    #         try:
    #             new_cards = self.deck.deal(cards_to_discard)
    #         except ValueError:
    #             if self.verbose:
    #                 print("Ran out of cards to draw")
    #             num_discards = np.nan
    #             return num_discards
    #         self.update(new_cards)
    #         current_suits = self.get_suit_count(self.hand, count_dict={})


    #         num_discards += 1

    #     if self.verbose:
    #         print(f"It took {num_discards} discards to achieve a flush")
    #     return num_discards

    # def simulate_greedy_straight(self):
    #     """Simulate one round with the goal of getting a straight using the greedy method"""
    #     # Does not value higher straights
        

    #     num_discards = 0
    #     current_ranks = self.obs_ranks
    #     while not self.is_straight(current_ranks):
    #         # Detect which straight is most likely
    #         hand_rank_set = set(current_ranks.keys())
    #         straight_references_hits = [len(s.intersection(hand_rank_set)) for s in straight_references]
    #         target_straight_ind = np.argmax(straight_references_hits[::-1])
    #         target_straight = straight_references[-(target_straight_ind+1)] # target the highest viable straight

    #         # keep only the values in the target straight
    #         hand_to_keep = [
    #             card for card in self.hand 
    #             if self.RANKS[card.rank] in target_straight
    #             ] # filter to only keep ranks in target straight
    #         current_ranks = self.get_rank_count(hand_to_keep, count_dict={}) # get rank count to check for duplicates
    #         max_rank_count = max(current_ranks.values())
    #         hand_to_keep_has_duplicates = max_rank_count > 1
    #         if hand_to_keep_has_duplicates:
    #             # for each duplicate rank, pick a random card
    #             duplicate_ranks_in_hand_to_keep = [rank for rank, count in current_ranks.items() if count == max_rank_count] # identified duplicate ranks
    #             duplicate_rank_to_keep = [
    #                 next(card for card in hand_to_keep if self.RANKS[card.rank] == rank) # picking one card of a given rank 
    #                 for rank in duplicate_ranks_in_hand_to_keep
    #                 ]
    #             # recompose hand to keep as unique ranks + deduplicated ranks
    #             hand_to_keep = [card for card in hand_to_keep if self.RANKS[card.rank] not in duplicate_ranks_in_hand_to_keep] + duplicate_rank_to_keep
            
    #         cards_to_discard = self.starting_hand_size - len(hand_to_keep)
    #         discarding_too_many_cards = cards_to_discard > self.HAND_SIZE
    #         if discarding_too_many_cards:
    #             # need to add a random card back
    #             #TODO: what happens if I need to grab more than 1 card?
    #             random_card = next(card for card in self.hand if card not in hand_to_keep)
    #             hand_to_keep.append(random_card)

    #         self.hand = hand_to_keep
    #         cards_to_discard = self.starting_hand_size - len(self.hand)
    #         discarded_more_than_hand_size = cards_to_discard > self.HAND_SIZE
    #         if discarded_more_than_hand_size:
    #             raise ValueError(f"Error: Discarded more than {self.HAND_SIZE} cards is not allowed. Ended up discarding {cards_to_discard} cards")

    #         try:
    #             new_cards = self.deck.deal(cards_to_discard)
    #         except ValueError:
    #             if self.verbose:
    #                 print("Ran out of cards to draw")
    #             num_discards = np.nan
    #             return num_discards
    #         self.update(new_cards)
    #         current_suits = self.get_rank_count(self.hand, count_dict={})
    #         num_discards += 1

    #     if self.verbose:
    #         print(f"It took {num_discards} discards to achieve a straight")
    #     return num_discards

    # def simulate_likely_flush(self):
    #     """Simulate one round with the goal"""
    #     straight_references = [set(['A','2', '3', '4', '5'])] + [set(self.RANKS[i:i+5]) for i in range(9)]

    #     num_discards = 0
    #     current_suits = self.obs_suits
    #     start = time.time()
    #     while not self.is_flush(current_suits):
    #         suit_likelihood = self.get_likelihood(current_suits, hand_type="flush")
    #         max_likelihood = max(suit_likelihood.values())
    #         most_likely_suits = [suit for suit, likelihood in suit_likelihood.items() if likelihood == max_likelihood]

    #         suit_to_keep = np.random.choice(most_likely_suits).item() # randomly pick
    #         hand_to_keep = [
    #             card for card in self.hand
    #             if self.SUITS[card.suit]==suit_to_keep
    #             ] # filter to keep only target suit
            
    #         if len(hand_to_keep) <= self.min_suit_count:
    #             random_card = next(card for card in self.hand if self.SUITS[card.suit]!=suit_to_keep)
    #             hand_to_keep += [random_card]

    #         cards_to_discard = self.starting_hand_size - len(hand_to_keep)
    #         discarded_more_than_hand_size = cards_to_discard > self.HAND_SIZE
    #         if discarded_more_than_hand_size:
    #             raise ValueError(f"Error: Discarded can only be up to {self.HAND_SIZE} cards, but ended up discarding {cards_to_discard} cards")
            
    #         try:
    #             new_cards = self.deck.deal(cards_to_discard)
    #         except ValueError:
    #             if self.verbose:
    #                 print("Ran out of cards to draw")
    #             num_discards = np.nan
    #             return num_discards
    #         if num_discards > 52:
    #             raise ValueError("Impossible number of discards")

    #         self.hand = hand_to_keep
    #         self.update(new_cards)
    #         current_suits = self.get_suit_count(self.hand, count_dict={})
    #         num_discards += 1
    #         if time.time() - start > 60:
    #             raise ValueError()

    #     if self.verbose:
    #         print(f"It took {num_discards} discards to achieve a flush")
    #     return num_discards

    # def simulate_likely_straight(self):
        
    #     num_discards = 0
    #     current_ranks = self.obs_ranks
    #     start = time.time()
    #     while not self.is_straight(current_ranks):
    #         straight_likelihood = self.get_likelihood(current_ranks, hand_type="straight")
    #         max_likelihood = max(straight_likelihood.values())
    #         if max_likelihood == 0:
    #             return np.nan
    #         most_likely_straight = [straight for straight, likelihood in straight_likelihood.items() if likelihood == max_likelihood]
    #         target_straight = most_likely_straight[-1] # randomly highest ranked straight
    #         if self.verbose:
    #             print(target_straight, straight_likelihood[target_straight])

    #         # keep only the values in the target straight
    #         hand_to_keep = [
    #             card for card in self.hand 
    #             if self.RANKS[card.rank] in target_straight.split(",")
    #             ] # filter to only keep ranks in target straight
    #         if hand_to_keep == []:
    #             raise ValueError("Selected straight with no viable cards in hand")
    #         current_ranks = self.get_rank_count(hand_to_keep, count_dict={}) # get rank count to check for duplicates
    #         max_rank_count = max(current_ranks.values())
    #         hand_to_keep_has_duplicates = max_rank_count > 1
    #         if hand_to_keep_has_duplicates:
    #             # for each duplicate rank, pick a random card
    #             duplicate_ranks_in_hand_to_keep = [rank for rank, count in current_ranks.items() if count == max_rank_count] # identified duplicate ranks
    #             duplicate_rank_to_keep = [
    #                 next(card for card in hand_to_keep if self.RANKS[card.rank] == rank) # picking one card of a given rank 
    #                 for rank in duplicate_ranks_in_hand_to_keep
    #                 ]
                
    #             # recompose hand to keep as unique ranks + deduplicated ranks
    #             hand_to_keep = [card for card in hand_to_keep if self.RANKS[card.rank] not in duplicate_ranks_in_hand_to_keep] + duplicate_rank_to_keep
            
    #         cards_to_discard = self.starting_hand_size - len(hand_to_keep)
    #         discarding_too_many_cards = cards_to_discard > self.HAND_SIZE
    #         if discarding_too_many_cards:
    #             # need to add a random card back
    #             num_cards_to_add_back = cards_to_discard - self.HAND_SIZE
    #             cards_not_in_hand = iter([card for card in self.hand if card not in hand_to_keep])
    #             random_cards = [next(cards_not_in_hand, None) for _ in range(num_cards_to_add_back)]
    #             hand_to_keep += random_cards

            
    #         cards_to_discard = self.starting_hand_size - len(hand_to_keep)
    #         discarded_more_than_hand_size = cards_to_discard > self.HAND_SIZE
    #         if discarded_more_than_hand_size:
    #             raise ValueError(f"Error: Discarded more than {self.HAND_SIZE} cards is not allowed. Ended up discarding {cards_to_discard} cards")

    #         try:
    #             new_cards = self.deck.deal(cards_to_discard)
    #         except ValueError:
    #             if self.verbose:
    #                 print("Ran out of cards to draw")
    #             num_discards = np.nan
    #             return num_discards
    #         self.hand = hand_to_keep
    #         self.update(new_cards)
    #         current_suits = self.get_rank_count(self.hand, count_dict={})
    #         num_discards += 1
    #     if self.verbose:
    #         print(f"It took {num_discards} discards to achieve a straight")
    #     return num_discards

    def get_likelihood(self, current_cards, hand_type):
        """Calculate the chances of getting the cards needed to complete the hand in the next discard"""

        chances = {}
        
        if hand_type == "flush":
            observed_cards = self.obs_suits
        elif hand_type == "straight":
            observed_cards = self.obs_ranks
        else:
            raise ValueError("Unexpected hand type")

        total_seen_cards = sum(observed_cards.values())
        num_cards_left_in_deck = 52 - total_seen_cards
        if hand_type == "flush":
            # array of suit related numbers
            num_suits_in_hand_arr = np.array([current_cards.get(suit, 0) for suit in self.SUITS])
            num_suits_remaining_arr = np.array([13 - observed_cards.get(suit,0) for suit in self.SUITS])
            num_cards_to_draw_arr = np.minimum(self.starting_hand_size - num_suits_in_hand_arr, self.HAND_SIZE)
            num_cards_needed_arr = np.minimum(self.HAND_SIZE - num_suits_in_hand_arr, self.HAND_SIZE)
            chances_vals = 1 - hypergeom.cdf(
                num_cards_needed_arr - 1, 
                num_cards_left_in_deck, 
                num_suits_remaining_arr, # need to change, difficult as each rank is a different type of success while still neededing to hit all the needed cards
                num_cards_to_draw_arr
                )
            chances = dict(zip(self.SUITS, chances_vals))

        elif hand_type == "straight":
            # straight type x sorted values in straight
            num_straights_in_hand_arr = np.array([
                [current_cards.get(rank,0) for rank in straight] 
                for straight in self.straight_references])
            num_cards_per_straight = np.count_nonzero(num_straights_in_hand_arr, axis=1)
            ranks_needed_to_draw_mask = np.where(num_straights_in_hand_arr == 0, 1, 0)
            ranks_remaining_per_straight_arr = 4 - np.array([
                [observed_cards.get(rank, 0) for rank in straight]
                for straight in self.straight_references])
            needed_ranks_remaining_per_straight_arr = ranks_remaining_per_straight_arr * ranks_needed_to_draw_mask # keep only ranks that are needed
            num_cards_to_draw_arr = np.minimum(self.starting_hand_size - num_cards_per_straight, self.HAND_SIZE)
            num_cards_needed_arr = np.minimum(self.HAND_SIZE - num_cards_per_straight, self.HAND_SIZE)
            
            # problem
            """
            If I am aiming for a 5,6,7,8,9 straight and I need a 5 and 9, then I need to treat 5 and 9 as different pools.
            drawing another 5 cards means that hands need to have at least one 5 and at least one 9
            """
            # approximate by assuming all ranks are independent, true answer would use inclusion-exclusion principle
            rank_chances_vals = 1 - hypergeom.cdf(
                0, 
                num_cards_left_in_deck, # int
                needed_ranks_remaining_per_straight_arr, # matrix of straight type vs rank in straight where values are number needed and left in the deck
                num_cards_to_draw_arr[:, None]
                )
            rank_chances_vals_no_zeros = np.where(needed_ranks_remaining_per_straight_arr > 0, rank_chances_vals, 1) # replace all zeros with ones
            chances_vals = rank_chances_vals_no_zeros.prod(axis=1) # approximate by multiplying all together
            chances = dict(zip([",".join(straight) for straight in self.straight_references], chances_vals))

            # check for impossible to build straights
            straights_with_nans = np.isnan(rank_chances_vals).any(axis=1)
            if straights_with_nans.any():
                chances = {
                    straight: 0 if straights_with_nans[i] else likelihood for i, (straight, likelihood) in enumerate(chances.items())
                }

            not_enough_cards = (num_straights_in_hand_arr + needed_ranks_remaining_per_straight_arr>0).sum(axis=1) < 5
            if np.any(not_enough_cards):
                chances = {
                    straight: 0 if not_enough_cards[i] else likelihood for i, (straight, likelihood) in enumerate(chances.items())
                }
                


        else:
            raise ValueError("Unexpected hand type")
        

        has_nans = np.isnan(np.array(list(chances.values()))).any()
        if has_nans:
            raise ValueError("Detected nans in probabilities")
        elif all(val == 1.0 for val in chances.values()):
            raise ValueError("Sus")
        
        hand_not_in_max_prob = set(current_cards.keys()).intersection(set(max(chances, key=chances.get).split(","))) == {}
        if hand_not_in_max_prob:
            raise ValueError("Why is the max prob straight not in the hand?")

        return chances

    def raise_hand_type_error(self, hand_type):
        raise ValueError(f"Unexpected hand type. Expected 'flush' or 'straight'. Received {hand_type}")

    def raise_wrong_hand_type_error(self, hand_eval, hand_type):
        raise ValueError(f"Incompatible poker hand with hand_eval. {hand_eval} method does not work with {hand_type}")


def main():
    #print(run_simulation("likely_flush", num_simulations=1000))

    compare_methods()

    print()


def run_simulation(method, num_simulations=1000, verbose=False):
    """"""

    starting_hand_size = 8
    list_of_discards = []
    for _ in range(num_simulations):
        # Instantiate new deck
        deck = eval7.Deck()
        deck.shuffle()
        starting_hand = deck.deal(starting_hand_size)
        PH = PokerHands(starting_hand, deck, starting_hand_size, verbose=verbose)


        if method == "greedy_flush":
            num_discards = PH.simulate_one_round("flush", PH.get_greedy_flush, PH.is_flush)
        elif method == "likely_flush":
            num_discards = PH.simulate_one_round("flush", PH.get_likely_flush, PH.is_flush)
        elif method == "greedy_straight":
            num_discards = PH.simulate_one_round("straight", PH.get_greedy_straight, PH.is_straight)
        elif method == "likely_straight":
            num_discards = PH.simulate_one_round("straight", PH.get_likely_straight, PH.is_straight)
        else:
            raise ValueError("Unexpected method")
        list_of_discards.append(num_discards)
    
    return list_of_discards

def compare_methods():
    num_simulations = 100_000
    flush_discards_greedy = run_simulation("greedy_flush", num_simulations, verbose=False)
    print(f"{num_simulations} simulations got flushes with {np.nanmean(flush_discards_greedy)} discards on average")
    print(f"\tThere were {np.isnan(flush_discards_greedy).sum().item()} rounds that ran out of cards in the deck")

    flush_discards_likely = run_simulation("likely_flush", num_simulations, verbose=False)
    print(f"{num_simulations} simulations got flushes with {np.nanmean(flush_discards_likely)} discards on average")
    print(f"\tThere were {np.isnan(flush_discards_likely).sum().item()} rounds that ran out of cards in the deck")

    straight_discards_greedy = run_simulation("greedy_straight", num_simulations, verbose=False)
    print(f"{num_simulations} simulations got straights with {np.nanmean(straight_discards_greedy)} discards on average")
    print(f"\tThere were {np.isnan(straight_discards_greedy).sum().item()} rounds that ran out of cards in the deck")

    straight_discards_likely = run_simulation("likely_straight", num_simulations, verbose=False)
    print(f"{num_simulations} simulations got straights with {np.nanmean(straight_discards_likely)} discards on average")
    print(f"\tThere were {np.isnan(straight_discards_likely).sum().item()} rounds that ran out of cards in the deck")

    # Plotting
    fig, axs = plt.subplots(2, 2)

    axs[0,0].hist(flush_discards_greedy)
    axs[0,0].set_title("Num discards for Greedy Flushes")

    axs[0,1].hist(straight_discards_greedy)
    axs[0,1].set_title("Num discards for Greedy Straights")

    axs[1,0].hist(flush_discards_likely)
    axs[1,0].set_title("Num discards for Likely Flushes")

    axs[1,1].hist(straight_discards_likely)
    axs[1,1].set_title("Num discards for Likely Straights")

    plt.show()
    # print()

if __name__ == "__main__":
    main()
