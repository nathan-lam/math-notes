"""
Simulation to determine whether straights or flushes are easier to get in the game Balatro

A starting had has 8 cards, if you can discard and then draw up to 5 cards, how common is it to complete a flush vs straight
"""
import eval7, pprint
import numpy as np
from scipy.special import comb
from scipy.stats import hypergeom
import matplotlib.pyplot as plt

import time
from timer import time_methods

#@time_methods
class PokerHands:

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
    
    def is_flush(self, current_suits=None):
        """Detects if hand has a flush"""
        evaluated_hand = eval7.evaluate(self.hand)
        is_flush = (eval7.handtype(evaluated_hand) == "Flush") or (eval7.handtype(evaluated_hand) == "Straight Flush")
        return is_flush
    
    def is_straight(self, current_ranks=None):
        """Detects if hand has a straight"""
        evaluated_hand = eval7.evaluate(self.hand)
        is_straight = eval7.handtype(evaluated_hand) == "Straight"
        return is_straight

    def simulate_greedy_flush(self):
        """Simulate one round with the goal of getting a flush using the greedy method"""
        # Method only goes for the most abundant suit in hand, not what is most likely to be drawn

        num_discards = 0
        current_suits = self.obs_suits
        while not self.is_flush(current_suits):
            # Deciding which suit to keep
            max_count = max(current_suits.values())
            most_common_suits = [suit for suit, count in current_suits.items() if count == max_count]

            # Removing cards and then updating
            suit_to_keep = np.random.choice(most_common_suits).item() # randomly pick
            hand_to_keep = [
                card for card in self.hand
                if self.SUITS[card.suit]==suit_to_keep
                ] # filter to keep only target suit
            if len(hand_to_keep) <= self.min_suit_count:
                # if most common suit has 2 counts, then we need to keep 1 more card to satisfy hand size
                # only happens if suit count is 2-2-2-2 and hand_size is 8
                random_card = next(card for card in self.hand if self.SUITS[card.suit]!=suit_to_keep)
                hand_to_keep += [random_card]
            self.hand = hand_to_keep

            cards_to_discard = self.starting_hand_size - len(self.hand)
            discarded_more_than_hand_size = cards_to_discard > self.HAND_SIZE
            if discarded_more_than_hand_size:
                raise ValueError(f"Error: Discarded more than {self.HAND_SIZE} cards, but ended up discarding {cards_to_discard} cards")
            
            try:
                new_cards = self.deck.deal(cards_to_discard)
            except ValueError:
                if self.verbose:
                    print("Ran out of cards to draw")
                num_discards = np.nan
                return num_discards
            self.update(new_cards)
            current_suits = self.get_suit_count(self.hand, count_dict={})


            num_discards += 1

        if self.verbose:
            print(f"It took {num_discards} discards to achieve a flush")
        return num_discards

    def simulate_greedy_straight(self):
        """Simulate one round with the goal of getting a straight using the greedy method"""
        # Does not value higher straights
        straight_references = [set(['A','2', '3', '4', '5'])] + [set(self.RANKS[i:i+5]) for i in range(9)]

        num_discards = 0
        current_ranks = self.obs_ranks
        while not self.is_straight(current_ranks):
            # Detect which straight is most likely
            hand_rank_set = set(current_ranks.keys())
            straight_references_hits = [len(s.intersection(hand_rank_set)) for s in straight_references]
            target_straight_ind = np.argmax(straight_references_hits[::-1])
            target_straight = straight_references[-(target_straight_ind+1)] # target the highest viable straight

            # keep only the values in the target straight
            hand_to_keep = [
                card for i, card in enumerate(self.hand) 
                if self.RANKS[card.rank] in target_straight
                ] # filter to only keep ranks in target straight
            current_ranks = self.get_rank_count(hand_to_keep, count_dict={}) # get rank count to check for duplicates
            max_rank_count = max(current_ranks.values())
            hand_to_keep_has_duplicates = max_rank_count > 1
            if hand_to_keep_has_duplicates:
                # for each duplicate rank, pick a random card
                duplicate_ranks_in_hand_to_keep = [rank for rank, count in current_ranks.items() if count == max_rank_count] # identified duplicate ranks
                duplicate_rank_to_keep = [
                    next(card for card in hand_to_keep if self.RANKS[card.rank] == rank) # picking one card of a given rank 
                    for rank in duplicate_ranks_in_hand_to_keep
                    ]
                # recompose hand to keep as unique ranks + deduplicated ranks
                hand_to_keep = [card for card in hand_to_keep if self.RANKS[card.rank] not in duplicate_ranks_in_hand_to_keep] + duplicate_rank_to_keep
            
            cards_to_discard = self.starting_hand_size - len(hand_to_keep)
            discarding_too_many_cards = cards_to_discard > self.HAND_SIZE
            if discarding_too_many_cards:
                # need to add a random card back
                #TODO: what happens if I need to grab more than 1 card?
                random_card = next(card for card in self.hand if card not in hand_to_keep)
                hand_to_keep.append(random_card)

            self.hand = hand_to_keep
            cards_to_discard = self.starting_hand_size - len(self.hand)
            discarded_more_than_hand_size = cards_to_discard > self.HAND_SIZE
            if discarded_more_than_hand_size:
                raise ValueError(f"Error: Discarded more than {self.HAND_SIZE} cards is not allowed. Ended up discarding {cards_to_discard} cards")

            try:
                new_cards = self.deck.deal(cards_to_discard)
            except ValueError:
                if self.verbose:
                    print("Ran out of cards to draw")
                num_discards = np.nan
                return num_discards
            self.update(new_cards)
            current_suits = self.get_rank_count(self.hand, count_dict={})
            num_discards += 1

        if self.verbose:
            print(f"It took {num_discards} discards to achieve a straight")
        return num_discards

    def simulate_likely_flush(self):
        """Simulate one round with the goal"""
        
        num_discards = 0
        current_suits = self.obs_suits
        start = time.time()
        while not self.is_flush(current_suits):
            suit_likelihood = self.get_likelihood(current_suits, hand_type="flush")
            max_likelihood = max(suit_likelihood.values())
            most_likely_suits = [suit for suit, likelihood in suit_likelihood.items() if likelihood == max_likelihood]

            suit_to_keep = np.random.choice(most_likely_suits).item() # randomly pick
            hand_to_keep = [
                card for card in self.hand
                if self.SUITS[card.suit]==suit_to_keep
                ] # filter to keep only target suit
            
            if len(hand_to_keep) <= self.min_suit_count:
                random_card = next(card for card in self.hand if self.SUITS[card.suit]!=suit_to_keep)
                hand_to_keep += [random_card]

            cards_to_discard = self.starting_hand_size - len(hand_to_keep)
            discarded_more_than_hand_size = cards_to_discard > self.HAND_SIZE
            if discarded_more_than_hand_size:
                raise ValueError(f"Error: Discarded can only be up to {self.HAND_SIZE} cards, but ended up discarding {cards_to_discard} cards")
            
            try:
                new_cards = self.deck.deal(cards_to_discard)
            except ValueError:
                if self.verbose:
                    print("Ran out of cards to draw")
                num_discards = np.nan
                return num_discards
            if num_discards > 52:
                raise ValueError("Impossible number of discards")

            self.hand = hand_to_keep
            self.update(new_cards)
            current_suits = self.get_suit_count(self.hand, count_dict={})
            num_discards += 1
            if time.time() - start > 60:
                raise ValueError()

        if self.verbose:
            print(f"It took {num_discards} discards to achieve a flush")
        return num_discards

    def get_likelihood(self, current_suits, hand_type):
        """Calculate the chances of getting the cards needed to complete the hand in the next discard"""

        chances = {}
        observed_cards = self.obs_suits
        total_seen_cards = sum(observed_cards.values())
        num_cards_left_in_deck = 52 - total_seen_cards
        if hand_type == "flush":
            # array of suit related numbers
            num_suits_in_hand_arr = np.array([current_suits.get(suit, 0) for suit in self.SUITS])
            num_suits_remaining_arr = np.array([13 - observed_cards.get(suit,0) for suit in self.SUITS])
            num_cards_to_draw_arr = draw_arr = np.minimum(self.starting_hand_size - num_suits_in_hand_arr, self.HAND_SIZE)
            num_cards_needed_arr = np.minimum(self.HAND_SIZE - num_suits_in_hand_arr, self.HAND_SIZE)
            chances_vals = 1 - hypergeom.cdf(
                num_cards_needed_arr - 1, 
                num_cards_left_in_deck, 
                num_suits_remaining_arr, 
                num_cards_to_draw_arr
                )
            chances = dict(zip(self.SUITS, chances_vals))

        elif hand_type == "straight":
            pass
        else:
            raise ValueError("Unexpected hand type")
        
        return chances

    def hypergeo_prob(self, population_size, num_successes, num_cards_drawn, num_obs_successes):
        """Returns hypergeoemtric likelihood. Cards drawn follow a hyper geometric distribution"""
        sample_space = comb(population_size, num_cards_drawn)
        prob = (comb(num_successes, num_obs_successes) * comb(population_size-num_successes, num_cards_drawn - num_obs_successes)) / sample_space
        return prob

    def reset(self):
        self.hand = []
        self.obs_suits = {}
        self.obs_ranks = {}

class TestDeck:
    def __init__(self, test_mode=False):
        self.test_mode = test_mode

    def shuffle(self):
        pass

    def deal(self):
        pass


def main():
    starting_hand_size = 8

    deck = eval7.Deck()
    deck.shuffle()
    starting_hand = deck.deal(starting_hand_size)
    PH = PokerHands(starting_hand, deck, starting_hand_size)
    #print(PH.simulate_one_flush_round())
    #print(PH.simulate_one_straight_round())
    print(PH.simulate_prob_flush())
    print()

def simulation(method, num_simulations=1000, verbose=False):


    starting_hand_size = 8
    list_of_discards = []
    for _ in range(num_simulations):
        deck = eval7.Deck()
        deck.shuffle()
        starting_hand = deck.deal(starting_hand_size)
        PH = PokerHands(starting_hand, deck, starting_hand_size, verbose=verbose)


        if method == "greedy_flush":
            run_sim = PH.simulate_greedy_flush
        elif method == "likely_flush":
            run_sim = PH.simulate_likely_flush
        elif method == "greedy_straight":
            run_sim = PH.simulate_greedy_straight
        elif method == "likely_straight":
            run_sim = PH.simulate_likely_straight
        else:
            raise ValueError("Unexpected method")
        list_of_discards.append(run_sim())
    
    return list_of_discards

if __name__ == "__main__":
    # main()

    num_simulations = 100_000
    flush_discards_greedy = simulation("greedy_flush", num_simulations, verbose=False)
    print(f"{num_simulations} simulations got flushes with {np.nanmean(flush_discards_greedy)} discards on average")
    print(f"\tThere were {np.isnan(flush_discards_greedy).sum().item()} rounds that ran out of cards in the deck")

    flush_discards_likely = simulation("likely_flush", num_simulations, verbose=False)
    print(f"{num_simulations} simulations got flushes with {np.nanmean(flush_discards_likely)} discards on average")
    print(f"\tThere were {np.isnan(flush_discards_likely).sum().item()} rounds that ran out of cards in the deck")

    # straight_discards = simulation("straight", num_simulations, verbose=False)
    # print(f"{num_simulations} simulations got straights with {np.nanmean(straight_discards)} discards on average")
    # print(f"\tThere were {np.isnan(straight_discards).sum().item()} rounds that ran out of cards in the deck")

    fig, axs = plt.subplots(1, 2)

    axs[0].hist(flush_discards_greedy)
    axs[0].set_title("Num discards for Flushes")
    axs[1].hist(flush_discards_likely)
    axs[1].set_title("Num discards for Straights")
    plt.show()

    # print()